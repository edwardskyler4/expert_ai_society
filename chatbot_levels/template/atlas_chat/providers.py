"""Dependency-free HTTP adapters for OpenAI Responses and Ollama."""
import base64
import json
import socket
import urllib.error
import urllib.request


class ProviderError(RuntimeError):
    pass


def open_request(url, payload=None, key=None, timeout=120):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(), headers=headers)
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        # Do not return arbitrary provider bodies or credentials to the browser.
        friendly = {
            401: "The model provider rejected the API key.",
            403: "This account cannot access the requested model or tool.",
            404: "The model or endpoint was not found. Check the configured model name.",
            429: "The provider rate limit or account quota was reached.",
        }
        raise ProviderError(friendly.get(exc.code, f"The provider returned HTTP {exc.code}.")) from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise ProviderError("Cannot connect to the model provider. Check that it is running and reachable.") from exc


def json_request(url, payload=None, key=None):
    with open_request(url, payload, key) as response:
        return json.load(response)


def sse_objects(response):
    """Parse SSE messages, including multi-line data and Unicode byte streams."""
    data = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                joined = "\n".join(data)
                data.clear()
                if joined == "[DONE]":
                    return
                yield json.loads(joined)
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data and "\n".join(data) != "[DONE]":
        yield json.loads("\n".join(data))


def openai_input(messages, image_getter):
    result = []
    for message in messages:
        if message["role"] == "assistant":
            result.append({"role": "assistant", "content": message["content"]})
            continue
        content = [{"type": "input_text", "text": message["content"]}]
        for value in message.get("images", []):
            image = image_getter(value)
            encoded = base64.b64encode(image["data"]).decode()
            content.append({"type": "input_image", "image_url": f"data:{image['mime']};base64,{encoded}"})
        result.append({"role": "user", "content": content})
    return result


def annotations(output):
    sources, artifacts = [], []
    for item in output:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            for annotation in content.get("annotations", []):
                if annotation.get("type") == "url_citation":
                    url = annotation.get("url", "")
                    if url.startswith(("https://", "http://")) and not any(s["url"] == url for s in sources):
                        sources.append({"kind": "web", "url": url, "title": annotation.get("title", url)})
                elif annotation.get("type") == "container_file_citation":
                    artifacts.append({
                        "file_id": annotation.get("file_id"), "container_id": annotation.get("container_id"),
                        "name": annotation.get("filename", "artifact"),
                    })
    return sources, artifacts


class Providers:
    def __init__(self, settings, image_getter):
        self.settings, self.image_getter = settings, image_getter

    def validate(self, provider, model):
        if provider not in ("openai", "ollama", "demo"):
            raise ValueError("Select OpenAI or Ollama.")
        if not isinstance(model, str) or not model.strip() or len(model) > 120:
            raise ValueError("A valid model name is required.")
        if provider == "openai" and not self.settings.openai_key:
            raise ProviderError("Set OPENAI_API_KEY in .env, then restart Atlas, to use OpenAI.")
        if provider == "demo" and not self.settings.allow_demo:
            raise ValueError("Demo mode is disabled.")

    def stream(self, provider, model, instructions, messages, options, run):
        if provider == "demo":
            text = (
                "**Interface demo — no AI model is connected.**\n\n"
                "Atlas can stream replies, preserve conversations, and retrieve uploaded documents. "
                "Select OpenAI or Ollama in Settings to receive real model answers."
            )
            for word in text.split(" "):
                if run.cancelled.is_set():
                    return
                yield {"type": "delta", "text": word + " "}
            yield {"type": "provider_done", "usage": {}, "sources": [], "artifacts": []}
            return
        if provider == "openai":
            yield from self._openai(model, instructions, messages, options, run)
        else:
            yield from self._ollama(model, instructions, messages, options, run)

    def _openai(self, model, instructions, messages, options, run):
        tools = []
        if options.get("web"):
            tools.append({"type": "web_search"})
        if options.get("code"):
            tools.append({"type": "code_interpreter", "container": {"type": "auto"}})
        payload = {
            "model": model, "instructions": instructions,
            "input": openai_input(messages, self.image_getter), "stream": True,
            "store": False, "max_output_tokens": options["max_tokens"],
        }
        if tools:
            payload["tools"] = tools
            payload["max_tool_calls"] = 6
        if options.get("reasoning"):
            payload["reasoning"] = {"effort": options["reasoning"]}
        with open_request(self.settings.openai_base + "/responses", payload, self.settings.openai_key) as response:
            run.upstream = response
            finished = False
            for event in sse_objects(response):
                if run.cancelled.is_set():
                    return
                kind = event.get("type", "")
                if kind == "response.output_text.delta":
                    yield {"type": "delta", "text": event["delta"]}
                elif kind == "response.refusal.delta":
                    yield {"type": "delta", "text": event["delta"]}
                elif kind in ("response.web_search_call.in_progress", "response.code_interpreter_call.in_progress"):
                    yield {"type": "status", "text": "Searching the web…" if "web_search" in kind else "Running Python in the hosted sandbox…"}
                elif kind in ("response.failed", "error"):
                    raise ProviderError("The model provider could not complete this response.")
                elif kind in ("response.completed", "response.incomplete"):
                    result = event.get("response", {})
                    sources, artifacts = annotations(result.get("output", []))
                    yield {
                        "type": "provider_done", "usage": result.get("usage", {}),
                        "sources": sources, "artifacts": artifacts,
                        "incomplete": kind == "response.incomplete",
                    }
                    finished = True
            if not finished and not run.cancelled.is_set():
                raise ProviderError("The provider stream ended before a completion event.")

    def _ollama(self, model, instructions, messages, options, run):
        history = [{"role": "system", "content": instructions}]
        for message in messages:
            item = {"role": message["role"], "content": message["content"]}
            if message.get("images"):
                item["images"] = [base64.b64encode(self.image_getter(value)["data"]).decode() for value in message["images"]]
            history.append(item)
        payload = {
            "model": model, "messages": history, "stream": True,
            "options": {"temperature": options["temperature"], "num_predict": options["max_tokens"]},
        }
        with open_request(self.settings.ollama_base + "/api/chat", payload) as response:
            run.upstream = response
            finished = False
            for raw in response:
                if run.cancelled.is_set():
                    return
                if not raw.strip():
                    continue
                event = json.loads(raw)
                if "error" in event:
                    raise ProviderError("Ollama reported an error. Check the installed model and its capabilities.")
                # Deliberately consume only public answer text, not private thinking tokens.
                text = event.get("message", {}).get("content", "")
                if text:
                    yield {"type": "delta", "text": text}
                if event.get("done"):
                    finished = True
                    yield {
                        "type": "provider_done", "sources": [], "artifacts": [],
                        "usage": {"input_tokens": event.get("prompt_eval_count", 0), "output_tokens": event.get("eval_count", 0)},
                        "incomplete": event.get("done_reason") == "length",
                    }
            if not finished and not run.cancelled.is_set():
                raise ProviderError("Ollama's stream ended before completion.")

    def embeddings(self, texts):
        provider, model = self.settings.embedding_provider, self.settings.embedding_name
        if provider == "none":
            return None
        if provider not in ("openai", "ollama"):
            raise ValueError("EMBEDDING_PROVIDER must be none, openai, or ollama.")
        vectors = []
        for start in range(0, len(texts), 16):
            batch = texts[start:start + 16]
            if provider == "openai":
                if not self.settings.openai_key:
                    raise ProviderError("OpenAI embeddings require OPENAI_API_KEY.")
                response = json_request(self.settings.openai_base + "/embeddings", {"model": model, "input": batch}, self.settings.openai_key)
                part = [item["embedding"] for item in sorted(response["data"], key=lambda item: item["index"])]
            else:
                response = json_request(self.settings.ollama_base + "/api/embed", {"model": model, "input": batch, "truncate": False})
                part = response["embeddings"]
            if len(part) != len(batch):
                raise ProviderError("The embedding provider returned an unexpected number of vectors.")
            vectors.extend(part)
        return vectors

    def artifact(self, container_id, file_id):
        if not self.settings.openai_key:
            raise ProviderError("Downloading this artifact requires OPENAI_API_KEY.")
        import re
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", container_id or "") or not re.fullmatch(r"[a-zA-Z0-9_-]+", file_id or ""):
            raise ValueError("Invalid artifact identifier.")
        with open_request(f"{self.settings.openai_base}/containers/{container_id}/files/{file_id}/content", key=self.settings.openai_key) as response:
            data = response.read(20 * 1024 * 1024 + 1)
        if len(data) > 20 * 1024 * 1024:
            raise ValueError("Artifact exceeds the 20 MiB download limit.")
        return data
