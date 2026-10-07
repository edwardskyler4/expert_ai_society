"""Optional live model smoke evaluations. Calls the selected provider, which may bill usage."""
import argparse
import json
from pathlib import Path
import tempfile

from engine import Engine
from providers import Providers
from retrieval import Retriever
from settings import Settings, load_dotenv
from store import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openai", "ollama"], required=True)
    parser.add_argument("--model")
    parser.add_argument("--out", type=Path, default=Path("evaluation-results.json"))
    args = parser.parse_args()
    load_dotenv(Path(__file__).parent / ".env")
    settings = Settings(embedding_provider="none")
    model = args.model or (settings.openai_model if args.provider == "openai" else settings.ollama_model)
    cases = [
        {"name":"document_fact", "prompt":"When is the Nimbus private beta scheduled? Give the date from the notes and cite the passage.", "needles":["16", "2026", "[D1]"]},
        {"name":"document_limit", "prompt":"According to the Nimbus notes, how many invited testers will the beta accept?", "needles":["25"]},
        {"name":"explicit_memory", "prompt":"What programming language do I prefer for code examples?", "needles":["python"]},
        {"name":"conversation_history", "first":"For this conversation, the secret project nickname is Marigold. Reply briefly.", "prompt":"What project nickname did I just give you?", "needles":["marigold"]},
    ]
    results = []
    with tempfile.TemporaryDirectory() as directory:
        store = Store(Path(directory) / "eval.sqlite3")
        providers = Providers(settings, store.image)
        providers.validate(args.provider, model)
        retriever = Retriever(store, providers, settings)
        retriever.ingest("project-notes.md", (Path(__file__).parent / "examples/project-notes.md").read_bytes())
        store.add_memory("I prefer Python for code examples.")
        engine = Engine(store, providers, retriever, settings)
        options = {"provider":args.provider, "model":model, "max_tokens":4096, "temperature":0}
        for case in cases:
            conversation = store.create_conversation()
            if case.get("first"):
                list(engine.events(engine.begin(conversation["id"], case["first"], options)))
            events = list(engine.events(engine.begin(conversation["id"], case["prompt"], options)))
            final = events[-1]
            answer = final["content"]
            passed = final["status"] == "complete" and all(needle in answer.lower() for needle in case["needles"])
            results.append({"name":case["name"],"passed":passed,"answer":answer,"status":final["status"],"metadata":final["metadata"]})
            print(f"{'PASS' if passed else 'FAIL'} {case['name']}")
    args.out.write_text(json.dumps({"provider":args.provider,"model":model,"results":results},indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"Saved {args.out}. These exact-string smoke checks do not measure general intelligence.")


if __name__ == "__main__":
    main()
