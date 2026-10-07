import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from app import build
from engine import BusyError, Engine, Run, bounded_history
from providers import ProviderError, Providers, annotations, openai_input, sse_objects
from retrieval import Retriever, chunks, cosine, extract, fuse
from server import image_type, safe_name
from settings import Settings, load_dotenv
from store import Store


class FakeProvider:
    def __init__(self, fail=False):
        self.fail = fail
        self.last_messages = None
        self.last_instructions = None

    def validate(self, provider, model):
        pass

    def embeddings(self, texts):
        return None

    def stream(self, provider, model, instructions, messages, options, run):
        self.last_messages, self.last_instructions = messages, instructions
        yield {"type": "delta", "text": "Hello "}
        if self.fail:
            raise ProviderError("Simulated failure.")
        yield {"type": "delta", "text": "world."}
        yield {"type": "provider_done", "usage": {"output_tokens": 2}, "sources": [], "artifacts": []}


class BaseCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.settings = Settings(data_dir=Path(self.temporary.name), openai_key="", allow_demo=True,
                                 embedding_provider="none")
        self.store = Store(self.settings.data_dir / "test.sqlite3")
        self.provider = FakeProvider()
        self.retriever = Retriever(self.store, self.provider, self.settings)
        self.engine = Engine(self.store, self.provider, self.retriever, self.settings)


class StorageTests(BaseCase):
    def test_turn_and_restart_preserve_history(self):
        conversation = self.store.create_conversation()
        _, assistant = self.store.begin_turn(conversation["id"], "First question", [])
        self.store.finish_message(assistant, "Partial answer", "streaming", {"sources": []})
        reopened = Store(self.store.path)
        result = reopened.conversation(conversation["id"])
        self.assertEqual(result["title"], "First question")
        self.assertEqual(result["messages"][-1]["status"], "interrupted")
        self.assertEqual(result["messages"][-1]["content"], "Partial answer")

    def test_delete_cascades_and_retains_shared_images(self):
        image = self.store.add_image("image.png", "image/png", b"image")
        a, b = self.store.create_conversation(), self.store.create_conversation()
        self.store.begin_turn(a["id"], "a", [image["id"]])
        self.store.begin_turn(b["id"], "b", [image["id"]])
        self.store.delete_conversation(a["id"])
        self.assertEqual(self.store.image(image["id"])["data"], b"image")
        self.store.delete_conversation(b["id"])
        with self.assertRaises(KeyError):
            self.store.image(image["id"])
        with self.store.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM messages").fetchone()[0], 0)

    def test_memory_is_explicit_and_deletable(self):
        value = self.store.add_memory("I prefer Python examples.")
        self.assertEqual(len(self.store.list_memories()), 1)
        self.store.delete_memory(value["id"])
        self.assertEqual(self.store.list_memories(), [])

    def test_deleting_chat_keeps_new_unattached_uploads(self):
        conversation = self.store.create_conversation()
        image = self.store.add_image("pending.png", "image/png", b"new upload")
        self.store.delete_conversation(conversation["id"])
        self.assertEqual(self.store.image(image["id"])["data"], b"new upload")

    def test_missing_image_rolls_back_turn(self):
        conversation = self.store.create_conversation()
        with self.assertRaises(ValueError):
            self.store.begin_turn(conversation["id"], "Hello", ["missing"])
        self.assertEqual(self.store.conversation(conversation["id"])["messages"], [])


class RetrievalTests(BaseCase):
    def test_chunks_cover_input_and_make_progress(self):
        source = "abcdefghijklmnopqrstuvwxyz" * 200
        result = chunks(source, 200, 30)
        self.assertTrue(all(0 < len(part) <= 200 for part in result))
        self.assertTrue(result[0].startswith(source[:10]))
        self.assertTrue(result[-1].endswith(source[-10:]))
        with self.assertRaises(ValueError):
            chunks(source, 10, 10)

    def test_bm25_returns_relevant_document(self):
        self.retriever.ingest("banana.txt", b"Bananas are yellow fruit. Banana bread is delicious.")
        self.retriever.ingest("space.txt", b"Jupiter is a planet in our solar system.")
        sources, warnings = self.retriever.search("What planet is Jupiter?")
        self.assertEqual(sources[0]["title"], "space.txt")
        self.assertEqual(sources[0]["label"], "D1")
        self.assertEqual(warnings, [])

    def test_delete_removes_fts_rows(self):
        document = self.retriever.ingest("space.txt", b"Jupiter is a planet.")
        self.store.delete_document(document["id"])
        self.assertEqual(self.retriever.search("Jupiter")[0], [])

    def test_search_handles_punctuation_and_empty_query(self):
        self.retriever.ingest("code.txt", b"Python decorators wrap functions.")
        self.assertTrue(self.retriever.search('Python " OR * (decorators)')[0])
        self.assertEqual(self.retriever.search("the and a")[0], [])

    def test_embedding_failure_does_not_create_partial_document(self):
        def fail(texts):
            raise ProviderError("Offline")
        self.provider.embeddings = fail
        with self.assertRaises(ProviderError):
            self.retriever.ingest("doc.txt", b"Hello there.")
        self.assertEqual(self.store.list_documents(), [])

    def test_hybrid_semantic_search_and_fallback(self):
        self.settings.embedding_provider = "ollama"
        self.provider.embeddings = lambda texts: [[1.0, 0.0] if "canine" in text or "dog" in text else [0.0, 1.0] for text in texts]
        self.retriever.ingest("animals.txt", b"A canine enjoys walks.")
        sources, _ = self.retriever.search("dog")
        self.assertEqual(sources[0]["title"], "animals.txt")
        self.provider.embeddings = lambda texts: (_ for _ in ()).throw(ProviderError("Offline"))
        sources, warnings = self.retriever.search("canine")
        self.assertTrue(sources)
        self.assertTrue(warnings)

    def test_vector_spaces_are_not_mixed(self):
        self.settings.embedding_provider = "ollama"
        self.settings.embedding_model = "first-model"
        self.provider.embeddings = lambda texts: [[1, 0] for text in texts]
        self.retriever.ingest("one.txt", b"A canine enjoys walks.")
        self.settings.embedding_model = "second-model"
        self.assertEqual(self.retriever.search("dog")[0], [])

    def test_rank_fusion_and_cosine(self):
        self.assertEqual(fuse([["a", "b"], ["b", "c"]])[0], "b")
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1)
        self.assertEqual(cosine([1], [1, 2]), -1)
        self.assertEqual(extract("text.md", b"\xef\xbb\xbfhello"), "hello")
        with self.assertRaises(ValueError):
            extract("bad.txt", b"\xff")

    def test_blank_pdf_has_clear_error_when_pdf_support_available(self):
        try:
            from pypdf import PdfWriter
        except ImportError:
            self.skipTest("Optional PDF support is not installed")
        writer, output = PdfWriter(), io.BytesIO()
        writer.add_blank_page(width=72, height=72)
        writer.write(output)
        with self.assertRaisesRegex(ValueError, "no extractable text"):
            extract("blank.pdf", output.getvalue())


class EngineTests(BaseCase):
    def test_stream_persists_answer_and_usage(self):
        conversation = self.store.create_conversation()
        self.store.add_memory("I prefer concise answers.")
        self.retriever.ingest("notes.txt", b"Jupiter has many moons.")
        run = self.engine.begin(conversation["id"], "Tell me about Jupiter", {"provider": "demo"})
        events = list(self.engine.events(run))
        self.assertEqual(events[-1]["type"], "done")
        message = self.store.conversation(conversation["id"])["messages"][-1]
        self.assertEqual(message["content"], "Hello world.")
        self.assertEqual(message["status"], "complete")
        self.assertEqual(message["metadata"]["usage"]["output_tokens"], 2)
        self.assertIn("concise", self.provider.last_instructions)
        self.assertIn("Jupiter has many moons", self.provider.last_messages[-1]["content"])

    def test_provider_failure_preserves_partial_text(self):
        self.provider.fail = True
        conversation = self.store.create_conversation()
        run = self.engine.begin(conversation["id"], "Hi", {"provider": "demo"})
        events = list(self.engine.events(run))
        self.assertEqual(events[-1]["status"], "error")
        self.assertEqual(events[-1]["content"], "Hello ")
        self.assertEqual(events[-1]["metadata"]["error"], "Simulated failure.")

    def test_cancel_preserves_partial_reply_and_unlocks(self):
        conversation = self.store.create_conversation()
        run = self.engine.begin(conversation["id"], "Hi", {"provider": "demo"})
        events = self.engine.events(run)
        while next(events)["type"] != "delta":
            pass
        self.assertTrue(self.engine.cancel(run.id))
        rest = list(events)
        self.assertEqual(rest[-1]["status"], "cancelled")
        self.assertEqual(rest[-1]["content"], "Hello ")
        self.assertFalse(run.lock.locked())

    def test_disconnect_releases_conversation(self):
        conversation = self.store.create_conversation()
        run = self.engine.begin(conversation["id"], "Hi", {"provider": "demo"})
        events = self.engine.events(run)
        next(events)
        events.close()
        self.assertFalse(run.lock.locked())
        self.assertEqual(self.store.conversation(conversation["id"])["messages"][-1]["status"], "cancelled")

    def test_duplicate_concurrent_turn_rejected(self):
        conversation = self.store.create_conversation()
        run = self.engine.begin(conversation["id"], "Hi", {"provider": "demo"})
        with self.assertRaises(BusyError):
            self.engine.begin(conversation["id"], "Another", {"provider": "demo"})
        self.engine.cancel(run.id)
        list(self.engine.events(run))

    def test_invalid_options_do_not_write_messages(self):
        conversation = self.store.create_conversation()
        for options in [{"max_tokens": 1}, {"reasoning": "invalid"}, {"provider": "ollama", "web": True}]:
            with self.assertRaises(ValueError):
                self.engine.begin(conversation["id"], "Hi", options)
        self.assertEqual(self.store.conversation(conversation["id"])["messages"], [])

    def test_context_preserves_pairs_and_latest_question(self):
        records = [
            {"role":"user", "content":"a"*20, "status":"complete"},
            {"role":"assistant", "content":"b"*20, "status":"complete"},
            {"role":"user", "content":"c"*20, "status":"complete"},
            {"role":"assistant", "content":"d"*20, "status":"complete"},
            {"role":"user", "content":"new", "status":"complete"},
            {"role":"assistant", "content":"", "status":"streaming"},
        ]
        kept, dropped = bounded_history(records, 50)
        self.assertEqual([record["content"] for record in kept], ["c"*20, "d"*20, "new"])
        self.assertEqual(dropped, 1)


class WireTests(unittest.TestCase):
    def test_sse_unicode_comments_and_multiline(self):
        stream = io.BytesIO(': heartbeat\ndata: {"text":\ndata: "café"}\n\ndata: [DONE]\n\n'.encode())
        self.assertEqual(list(sse_objects(stream)), [{"text":"café"}])

    def test_annotations_and_images(self):
        sources, artifacts = annotations([{"type":"message", "content":[{"annotations":[
            {"type":"url_citation", "url":"https://example.com", "title":"Example"},
            {"type":"url_citation", "url":"javascript:alert(1)", "title":"Unsafe"},
            {"type":"container_file_citation", "container_id":"cntr_one", "file_id":"file_one", "filename":"plot.png"},
        ]}]}])
        self.assertEqual(len(sources), 1)
        self.assertEqual(artifacts[0]["name"], "plot.png")
        result = openai_input([{"role":"user", "content":"Look", "images":["one"]}], lambda value: {"mime":"image/png", "data":b"abc"})
        self.assertEqual(result[0]["content"][1]["image_url"], "data:image/png;base64,YWJj")

    def test_filename_and_image_validation(self):
        self.assertEqual(safe_name("../../my notes.md"), "my notes.md")
        self.assertEqual(image_type(b"\x89PNG\r\n\x1a\nanything"), "image/png")
        with self.assertRaises(ValueError):
            image_type(b"<svg onload=alert(1)>")


class MockModelHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, body, self.headers.get("Authorization")))
        if self.path == "/v1/responses":
            payloads = [
                {"type":"response.output_text.delta", "delta":"A streamed answer."},
                {"type":"response.completed", "response":{"usage":{"input_tokens":12,"output_tokens":4},"output":[
                    {"type":"message","content":[{"annotations":[{"type":"url_citation","url":"https://example.com","title":"Example"}]}]}
                ]}},
            ]
            data = ''.join('data: '+json.dumps(event)+'\n\n' for event in payloads).encode()
            mime = "text/event-stream"
        elif self.path == "/api/chat":
            data = (json.dumps({"message":{"content":"Local reply."}, "done":False})+'\n'+json.dumps({"message":{"content":""},"done":True,"eval_count":3,"prompt_eval_count":8})+'\n').encode()
            mime = "application/x-ndjson"
        elif self.path == "/v1/embeddings":
            data = json.dumps({"data":[{"index":i,"embedding":[1.0,0.0]} for i,_ in enumerate(body["input"])]}).encode()
            mime = "application/json"
        elif self.path == "/api/embed":
            data = json.dumps({"embeddings":[[1.0,0.0] for _ in body["input"]]}).encode()
            mime = "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class ProviderHTTPTests(BaseCase):
    def setUp(self):
        super().setUp()
        self.mock = ThreadingHTTPServer(("127.0.0.1", 0), MockModelHandler)
        self.mock.requests = []
        threading.Thread(target=self.mock.serve_forever, daemon=True).start()
        self.addCleanup(self.mock.server_close)
        self.addCleanup(self.mock.shutdown)
        base = f"http://127.0.0.1:{self.mock.server_port}"
        self.settings.openai_base = base + "/v1"
        self.settings.ollama_base = base
        self.settings.openai_key = "test-key"
        self.providers = Providers(self.settings, self.store.image)

    def run_stream(self, provider):
        run = Run("run", "conversation", "user", "assistant", provider, "test-model", {}, threading.Lock())
        return list(self.providers.stream(provider, "test-model", "Instructions", [{"role":"user","content":"Hello","images":[]}],
                    {"max_tokens":1024,"temperature":0.7,"web":provider=="openai","code":provider=="openai","reasoning":""}, run))

    def test_openai_wire_format_tools_and_citations(self):
        events = self.run_stream("openai")
        self.assertEqual(events[0]["text"], "A streamed answer.")
        self.assertEqual(events[-1]["sources"][0]["url"], "https://example.com")
        path, body, authorization = self.mock.requests[-1]
        self.assertEqual(path, "/v1/responses")
        self.assertFalse(body["store"])
        self.assertEqual(authorization, "Bearer test-key")
        self.assertEqual([tool["type"] for tool in body["tools"]], ["web_search","code_interpreter"])

    def test_ollama_wire_format_and_usage(self):
        events = self.run_stream("ollama")
        self.assertEqual(events[0]["text"], "Local reply.")
        self.assertEqual(events[-1]["usage"]["output_tokens"], 3)
        self.assertEqual(self.mock.requests[-1][1]["messages"][0]["role"], "system")

    def test_embedding_adapters(self):
        for provider in ("openai", "ollama"):
            self.settings.embedding_provider = provider
            self.assertEqual(self.providers.embeddings(["a","b"]), [[1.0,0.0],[1.0,0.0]])

    def test_missing_key_and_disabled_demo_rejected(self):
        self.settings.openai_key = ""
        with self.assertRaises(ProviderError):
            self.providers.validate("openai", "model")
        self.settings.allow_demo = False
        with self.assertRaises(ValueError):
            self.providers.validate("demo", "model")


class APIHTTPTests(BaseCase):
    def setUp(self):
        super().setUp()
        self.server = build(self.settings, 0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def request(self, path, data=None, method=None, headers=None):
        request = urllib.request.Request(self.base+path, data=None if data is None else json.dumps(data).encode(),
                  method=method, headers={"Content-Type":"application/json", **(headers or {})})
        return urllib.request.urlopen(request, timeout=10)

    def test_full_conversation_stream_and_persistence(self):
        with self.request("/api/conversations", {}) as response:
            conversation = json.load(response)
        with self.request("/api/chat", {"conversation_id":conversation["id"],"text":"Hello","options":{"provider":"demo","model":"interface-demo"}}) as response:
            events = list(sse_objects(response))
        self.assertEqual(events[-1]["status"], "complete")
        with self.request("/api/conversations/"+conversation["id"]) as response:
            saved = json.load(response)
        self.assertEqual(saved["messages"][-1]["content"], events[-1]["content"])

    def test_upload_memory_delete_and_config_secret_redaction(self):
        with self.request("/api/documents", {"name":"notes.txt","data":base64.b64encode(b"Jupiter is large.").decode()}) as response:
            document = json.load(response)
        with self.request("/api/memories", {"content":"Use Python examples."}) as response:
            memory = json.load(response)
        with self.request("/api/config") as response:
            config = json.load(response)
        self.assertNotIn("openai_key", config)
        with self.request("/api/documents/"+document["id"], method="DELETE") as response:
            self.assertTrue(json.load(response)["ok"])
        with self.request("/api/memories/"+memory["id"], method="DELETE") as response:
            self.assertTrue(json.load(response)["ok"])

    def test_cross_origin_and_dns_rebinding_blocked(self):
        for headers in ({"Origin":"https://attacker.example"}, {"Host":"attacker.example"}):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.request("/api/conversations", {}, headers=headers)
            self.assertEqual(caught.exception.code, 403)

    def test_static_csp_and_unknown_route(self):
        with self.request("/") as response:
            self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
            self.assertIn(b"Make room for", response.read())
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/does-not-exist")
        self.assertEqual(caught.exception.code, 404)

    def test_bad_message_returns_error_without_persisting_turn(self):
        with self.request("/api/conversations", {}) as response:
            conversation = json.load(response)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/api/chat", {"conversation_id":conversation["id"],"text":"","options":{"provider":"demo","model":"demo"}})
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(self.server.store.conversation(conversation["id"])["messages"], [])


if __name__ == "__main__":
    unittest.main()
