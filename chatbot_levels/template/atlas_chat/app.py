"""Run Atlas: python app.py. Python 3.10+, no required packages."""
import argparse
from pathlib import Path

from engine import Engine
from providers import Providers
from retrieval import Retriever
from server import AtlasServer
from settings import Settings, load_dotenv
from store import Store


def build(settings, port=8000):
    store = Store(settings.data_dir / "atlas.sqlite3")
    providers = Providers(settings, store.image)
    retriever = Retriever(store, providers, settings)
    engine = Engine(store, providers, retriever, settings)
    return AtlasServer(("127.0.0.1", port), store, engine, providers, retriever, settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--demo", action="store_true", help="Enable an explicitly labeled interface demo without a model.")
    args = parser.parse_args()
    load_dotenv(Path(__file__).parent / ".env")
    settings = Settings()
    if args.demo:
        settings.allow_demo = True
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between 0 and 65535.")
    server = build(settings, args.port)
    print(f"Atlas is running at http://127.0.0.1:{server.server_port}")
    print("Chats, memory, and documents are stored in", settings.data_dir.resolve())
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Atlas.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
