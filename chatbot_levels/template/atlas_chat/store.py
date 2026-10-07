"""SQLite storage, transactions, and explicit long-term memory."""
import contextlib
import json
import sqlite3
import time
import uuid
from pathlib import Path


def identifier():
    return uuid.uuid4().hex


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL, content TEXT NOT NULL, status TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}', images TEXT NOT NULL DEFAULT '[]', created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_conversation ON messages(conversation_id, created);
                CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, content TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, characters INTEGER NOT NULL,
                    chunks INTEGER NOT NULL, embedding_signature TEXT NOT NULL, created REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL, text TEXT NOT NULL, vector TEXT
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    id UNINDEXED, document_id UNINDEXED, text, tokenize='unicode61'
                );
                CREATE TABLE IF NOT EXISTS images (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, mime TEXT NOT NULL, data BLOB NOT NULL, created REAL NOT NULL
                );
            """)
            # A restart cannot resume the previous HTTP streams.
            db.execute("UPDATE messages SET status='interrupted' WHERE status='streaming'")

    @contextlib.contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def create_conversation(self, title="New conversation"):
        value, now = identifier(), time.time()
        with self.connect() as db:
            db.execute("INSERT INTO conversations VALUES (?,?,?,?)", (value, title, now, now))
        return self.conversation(value)

    def list_conversations(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM conversations ORDER BY updated DESC")]

    def conversation(self, value):
        with self.connect() as db:
            row = db.execute("SELECT * FROM conversations WHERE id=?", (value,)).fetchone()
            if row is None:
                raise KeyError("Conversation not found.")
            result = dict(row)
            messages = db.execute("SELECT * FROM messages WHERE conversation_id=? ORDER BY created, rowid", (value,))
            result["messages"] = [self.message_dict(row) for row in messages]
            return result

    @staticmethod
    def message_dict(row):
        item = dict(row)
        item["metadata"] = json.loads(item["metadata"])
        item["images"] = json.loads(item["images"])
        return item

    def rename(self, value, title):
        with self.connect() as db:
            if not db.execute("UPDATE conversations SET title=?, updated=? WHERE id=?", (title, time.time(), value)).rowcount:
                raise KeyError("Conversation not found.")

    def delete_conversation(self, value):
        with self.connect() as db:
            former_images = {image for row in db.execute("SELECT images FROM messages WHERE conversation_id=?", (value,)) for image in json.loads(row[0])}
            if not db.execute("DELETE FROM conversations WHERE id=?", (value,)).rowcount:
                raise KeyError("Conversation not found.")
            # Remove this chat's attachments only when no surviving chat references them.
            # Keep newly uploaded images that the browser has not yet attached to a turn.
            used = {image for row in db.execute("SELECT images FROM messages") for image in json.loads(row[0])}
            for image in former_images - used:
                db.execute("DELETE FROM images WHERE id=?", (image,))

    def begin_turn(self, conversation_id, text, images):
        user_id, assistant_id, now = identifier(), identifier(), time.time()
        with self.connect() as db:
            current = db.execute("SELECT title FROM conversations WHERE id=?", (conversation_id,)).fetchone()
            if current is None:
                raise KeyError("Conversation not found.")
            for image in images:
                if db.execute("SELECT 1 FROM images WHERE id=?", (image,)).fetchone() is None:
                    raise ValueError("An image attachment was not found.")
            db.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?)",
                       (user_id, conversation_id, "user", text, "complete", "{}", json.dumps(images), now))
            db.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?)",
                       (assistant_id, conversation_id, "assistant", "", "streaming", "{}", "[]", now + 0.000001))
            title = text.replace("\n", " ")[:60] if current[0] == "New conversation" else current[0]
            db.execute("UPDATE conversations SET title=?, updated=? WHERE id=?", (title, now, conversation_id))
        return user_id, assistant_id

    def finish_message(self, value, text, status, metadata):
        with self.connect() as db:
            db.execute("UPDATE messages SET content=?, status=?, metadata=? WHERE id=?",
                       (text, status, json.dumps(metadata, ensure_ascii=False), value))

    def list_memories(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM memories ORDER BY created")]

    def add_memory(self, text):
        value, now = identifier(), time.time()
        with self.connect() as db:
            if db.execute("SELECT count(*) FROM memories").fetchone()[0] >= 50:
                raise ValueError("Memory is limited to 50 entries. Remove an entry first.")
            db.execute("INSERT INTO memories VALUES (?,?,?)", (value, text, now))
        return {"id": value, "content": text, "created": now}

    def delete_memory(self, value):
        with self.connect() as db:
            if not db.execute("DELETE FROM memories WHERE id=?", (value,)).rowcount:
                raise KeyError("Memory not found.")

    def list_documents(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM documents ORDER BY created DESC")]

    def delete_document(self, value):
        with self.connect() as db:
            db.execute("DELETE FROM chunks_fts WHERE document_id=?", (value,))
            if not db.execute("DELETE FROM documents WHERE id=?", (value,)).rowcount:
                raise KeyError("Document not found.")

    def add_image(self, name, mime, data):
        value = identifier()
        with self.connect() as db:
            db.execute("INSERT INTO images VALUES (?,?,?,?,?)", (value, name, mime, data, time.time()))
        return {"id": value, "name": name, "mime": mime}

    def image(self, value):
        with self.connect() as db:
            row = db.execute("SELECT * FROM images WHERE id=?", (value,)).fetchone()
            if row is None:
                raise KeyError("Image not found.")
            return dict(row)
