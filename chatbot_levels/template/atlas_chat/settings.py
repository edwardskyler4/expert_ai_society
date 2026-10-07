"""Server-owned configuration. Secrets never enter browser settings."""
import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path):
    if not Path(path).is_file():
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"\''))


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("ATLAS_DATA_DIR", "data")))
    openai_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-6-astra"))
    ollama_base: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen3:8b"))
    embedding_provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "none"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", ""))
    allow_demo: bool = field(default_factory=lambda: os.getenv("ATLAS_DEMO", "0") == "1")
    context_chars: int = 48000
    max_upload_bytes: int = 4 * 1024 * 1024
    max_document_chars: int = 400000

    @property
    def embedding_name(self):
        return self.embedding_model or (
            "text-embedding-3-small" if self.embedding_provider == "openai" else "embeddinggemma"
        )

    @property
    def embedding_signature(self):
        return f"{self.embedding_provider}:{self.embedding_name}"

    def public(self):
        return {
            "openai_configured": bool(self.openai_key),
            "openai_model": self.openai_model, "ollama_model": self.ollama_model,
            "embedding_provider": self.embedding_provider, "demo_enabled": self.allow_demo,
            "pdf_available": __import__("importlib.util", fromlist=["find_spec"]).find_spec("pypdf") is not None,
            "context_chars": self.context_chars,
        }
