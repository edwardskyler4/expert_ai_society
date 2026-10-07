"""A localhost-only HTTP API with streaming events and no required dependencies."""
import base64
import binascii
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit

from engine import BusyError
from providers import ProviderError, json_request

STATIC = Path(__file__).parent / "static"


def safe_name(name):
    if not isinstance(name, str):
        raise ValueError("A filename is required.")
    name = name.replace("\\", "/").split("/")[-1]
    return re.sub(r"[^\w .()\-]", "_", name)[:120] or "upload.txt"


def image_type(data):
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("Attach a PNG, JPEG, or WebP image.")


class AtlasServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, store, engine, providers, retriever, settings):
        super().__init__(address, Handler)
        self.store, self.engine, self.providers, self.retriever, self.settings = store, engine, providers, retriever, settings


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        # Paths and status codes only; never chat content or secrets.
        print(f"{self.address_string()} {format % args}")

    def headers_for(self, status, mime, length=None):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        if length is not None:
            self.send_header("Content-Length", str(length))

    def reply(self, value, status=200):
        payload = json.dumps(value, ensure_ascii=False).encode()
        self.headers_for(status, "application/json; charset=utf-8", len(payload))
        self.end_headers()
        self.wfile.write(payload)

    def local_request(self):
        host = self.headers.get("Host", "")
        expected = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if host not in expected:
            self.reply({"error": "Use the app's localhost address."}, 403)
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {"http://" + value for value in expected}:
            self.reply({"error": "Cross-origin requests are not allowed."}, 403)
            return False
        return True

    def body(self):
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            raise ValueError("Content-Type must be application/json.")
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= 6 * 1024 * 1024:
            raise ValueError("Request body is empty or exceeds the 6 MiB limit.")
        result = json.loads(self.rfile.read(length))
        if not isinstance(result, dict):
            raise ValueError("Request body must be a JSON object.")
        return result

    def dispatch(self, method):
        if not self.local_request():
            self.close_connection = True
            return
        parsed = urlsplit(self.path)
        path, query = parsed.path, parse_qs(parsed.query)
        try:
            self.route(method, path, query)
        except BusyError as exc:
            self.reply({"error": str(exc)}, 409)
        except KeyError as exc:
            self.reply({"error": exc.args[0]}, 404)
        except (ValueError, TypeError, binascii.Error) as exc:
            self.close_connection = True
            self.reply({"error": str(exc)}, 400)
        except ProviderError as exc:
            self.reply({"error": str(exc)}, 502)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            import traceback
            traceback.print_exc()
            self.reply({"error": "Internal server error. Check the server terminal."}, 500)

    def do_GET(self):
        self.dispatch("GET")

    def do_POST(self):
        self.dispatch("POST")

    def do_DELETE(self):
        self.dispatch("DELETE")

    def route(self, method, path, query):
        store, engine, settings = self.server.store, self.server.engine, self.server.settings
        if method == "GET" and path in ("/", "/app.js", "/styles.css"):
            file = STATIC / {"/": "index.html", "/app.js": "app.js", "/styles.css": "styles.css"}[path]
            data = file.read_bytes()
            mime = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
            self.headers_for(200, mime + "; charset=utf-8", len(data))
            self.end_headers()
            self.wfile.write(data)
            return
        if method == "GET" and path == "/api/config":
            self.reply(settings.public())
        elif method == "GET" and path == "/api/conversations":
            self.reply(store.list_conversations())
        elif method == "POST" and path == "/api/conversations":
            self.body()
            self.reply(store.create_conversation(), 201)
        elif path.startswith("/api/conversations/"):
            value = path.split("/")[-1]
            if method == "GET":
                self.reply(store.conversation(value))
            elif method in ("POST", "DELETE"):
                lock = engine.conversation_lock(value)
                if not lock.acquire(blocking=False):
                    raise BusyError("Stop the current response before changing this conversation.")
                try:
                    if method == "DELETE":
                        store.delete_conversation(value)
                    else:
                        title = self.body().get("title")
                        if not isinstance(title, str) or not title.strip() or len(title) > 100:
                            raise ValueError("Title must contain 1–100 characters.")
                        store.rename(value, title.strip())
                finally:
                    lock.release()
                self.reply({"ok": True})
            else:
                self.reply({"error": "Route not found."}, 404)
        elif method == "POST" and path == "/api/chat":
            body = self.body()
            options = body.get("options", {})
            if not isinstance(options, dict) or not isinstance(body.get("conversation_id"), str):
                raise ValueError("A conversation and settings object are required.")
            run = engine.begin(body["conversation_id"], body.get("text"), options, body.get("images", []))
            self.stream(run)
        elif method == "POST" and path == "/api/cancel":
            body = self.body()
            self.reply({"cancelled": engine.cancel(body.get("run_id", ""))})
        elif method == "GET" and path == "/api/documents":
            self.reply(store.list_documents())
        elif method == "POST" and path == "/api/documents":
            body = self.body()
            name = safe_name(body.get("name"))
            if Path(name).suffix.lower() not in (".txt", ".md", ".pdf", ".csv", ".json", ".py", ".js", ".ts", ".html", ".css", ".rst", ".log"):
                raise ValueError("Upload a text, Markdown, PDF, CSV, JSON, or source-code file.")
            data = base64.b64decode(body.get("data", ""), validate=True)
            if len(data) > settings.max_upload_bytes:
                raise ValueError("Uploads are limited to 4 MiB.")
            self.reply(self.server.retriever.ingest(name, data), 201)
        elif method == "DELETE" and path.startswith("/api/documents/"):
            store.delete_document(path.split("/")[-1])
            self.reply({"ok": True})
        elif method == "GET" and path == "/api/memories":
            self.reply(store.list_memories())
        elif method == "POST" and path == "/api/memories":
            text = self.body().get("content")
            if not isinstance(text, str) or not text.strip() or len(text) > 400:
                raise ValueError("Memory must contain 1–400 characters.")
            self.reply(store.add_memory(text.strip()), 201)
        elif method == "DELETE" and path.startswith("/api/memories/"):
            store.delete_memory(path.split("/")[-1])
            self.reply({"ok": True})
        elif method == "POST" and path == "/api/images":
            body = self.body()
            data = base64.b64decode(body.get("data", ""), validate=True)
            if len(data) > settings.max_upload_bytes:
                raise ValueError("Images are limited to 4 MiB.")
            self.reply(store.add_image(safe_name(body.get("name")), image_type(data), data), 201)
        elif method == "GET" and path.startswith("/api/images/"):
            image = store.image(path.split("/")[-1])
            self.headers_for(200, image["mime"], len(image["data"]))
            self.end_headers()
            self.wfile.write(image["data"])
        elif method == "POST" and path == "/api/connection":
            body = self.body()
            provider = body.get("provider")
            model = body.get("model") or (settings.openai_model if provider == "openai" else settings.ollama_model)
            self.server.providers.validate(provider, model)
            if provider == "demo":
                self.reply({"ok": True, "models": ["interface-demo"], "message": "Demo mode; no AI model is used."})
            elif provider == "ollama":
                result = json_request(settings.ollama_base + "/api/tags")
                models = [item["name"] for item in result.get("models", [])]
                self.reply({"ok": model in models or model + ":latest" in models, "models": models,
                            "message": "Ollama is reachable. Select an installed model."})
            else:
                from urllib.parse import quote
                json_request(settings.openai_base + "/models/" + quote(model, safe=""), key=settings.openai_key)
                self.reply({"ok": True, "models": [model], "message": "API key and model access verified. Tool support depends on the model."})
        elif method == "GET" and path == "/api/artifact":
            conversation = store.conversation(query.get("conversation", [""])[0])
            message = next((m for m in conversation["messages"] if m["id"] == query.get("message", [""])[0]), None)
            index = int(query.get("index", ["-1"])[0])
            artifacts = message["metadata"].get("artifacts", []) if message else []
            if not 0 <= index < len(artifacts):
                raise KeyError("Artifact not found.")
            item = artifacts[index]
            data = self.server.providers.artifact(item["container_id"], item["file_id"])
            name = safe_name(item["name"])
            self.headers_for(200, "application/octet-stream", len(data))
            self.send_header("Content-Disposition", f'attachment; filename="{name.encode("ascii", "replace").decode()}"')
            self.end_headers()
            self.wfile.write(data)
        else:
            self.reply({"error": "Route not found."}, 404)

    def stream(self, run):
        engine = self.server.engine
        events = engine.events(run)
        self.close_connection = True
        try:
            self.headers_for(200, "text/event-stream; charset=utf-8")
            self.send_header("Connection", "close")
            self.end_headers()
            for event in events:
                data = ("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode()
                self.wfile.write(data)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            run.cancelled.set()
        finally:
            events.close()
            if run.id in engine.runs:
                self.server.store.finish_message(run.assistant_id, "", "cancelled", {})
                engine.finish_run(run)
