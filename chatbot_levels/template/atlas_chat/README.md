# Atlas: a capable chatbot application

Atlas is a complete, local web application that connects a pretrained language
model to conversation history, saved memory, document retrieval, and optional
hosted tools. The application is written from scratch. The language model is
provided by OpenAI or Ollama, rather than trained from random weights.

The backend uses Python's standard library and SQLite. The frontend uses plain
HTML, CSS, and JavaScript. There is no required pip install, JavaScript build,
frontend framework, model-training job, or external database.

## Start the app

Use Python 3.10 or newer with SQLite FTS5 enabled. Extract the archive and open a
terminal inside `atlas_chat`:

```bash
cp .env.example .env
python3 app.py
```

Open **http://127.0.0.1:8000**. Click the model button at the top, choose a provider,
and test its connection. Configure one of the following paths for real AI replies.

### Hosted model: OpenAI

Edit `.env` with a project API key and a model your account can access:

```dotenv
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-6-astra
```

Restart the server, select OpenAI in Settings, and test the connection. The default
model comes from the current official documentation; you can change the name in
the app or `.env`. Availability and supported tools depend on your account and
selected model. Requests use the Responses API, with streaming and `store: false`.
API requests and hosted tools can incur provider charges. No live provider requests
were made during the creation of this project.

Web search and Python are per-request toggles. Turning a tool on makes it available
to the model; the model decides whether to use it. Python runs in OpenAI's hosted
container, not on your computer. Annotated files returned by that container appear
as download links, subject to provider retention and the app's 20 MiB download limit.
The app does not upload entire documents as sandbox files; selected retrieved text
is included in the model's prompt. Code-interpreter artifacts are not locally cached.

### Local model: Ollama

Install Ollama using its official instructions: https://ollama.com/download.
With the Ollama service running:

```bash
ollama pull qwen3:8b
python3 app.py
```

Select Ollama in Settings. If it is not already running as a background service,
run `ollama serve` in a separate terminal. The server uses `http://127.0.0.1:11434`
by default. Change `OLLAMA_BASE_URL` or `OLLAMA_MODEL` in `.env` if needed.

This route avoids hosted inference charges, but downloads model weights and uses
your hardware. Model quality, speed, memory requirements, and image support depend
on the model you install. The default Qwen model is for text. Choose a vision-capable
model if you attach images. Hosted web search and Python tools are only implemented
for OpenAI; local model tool-calling is not implemented.

### Interface demo without a model

```bash
python3 app.py --demo
```

Demo mode uses fixed, explicitly labeled text. It exercises the UI and stream
persistence but provides no AI answers. Enable it only when you want to inspect
the app without configuring a model. It is also used by the browser smoke test.

## Features

| Capability | Implementation |
| --- | --- |
| Streaming replies | POST request with server-sent events |
| Persistent conversations | SQLite messages, metadata, and image attachments |
| Saved memory | Explicit facts/preferences managed by the user |
| Document questions | Overlapping chunks, BM25 retrieval, source passages |
| Optional semantic search | Provider embeddings plus reciprocal-rank fusion |
| Image questions | PNG/JPEG/WebP attachments sent to a compatible model |
| Current-information lookup | OpenAI hosted web search and URL citations |
| Python computations | OpenAI hosted code interpreter and annotated file downloads |
| Model controls | Provider, model, output limit, reasoning effort, local temperature |
| Conversation management | Search titles, rename, delete, export Markdown |
| Long conversations | Complete-turn pruning with visible omission warnings |
| Recovery | Partial responses saved during streaming; restart marks unfinished replies |
| Stop generation | Cooperative cancellation and partial-answer preservation |
| Interface | Responsive layout, light/dark themes, safe Markdown/code rendering |

Memory is not silently extracted. Add, inspect, and delete entries in the Memory
panel. Each future request includes the saved entries. Deleting a memory does not
rewrite past messages that already used it.

Click the plus sign in the composer to attach up to four images. Image contents
are sent to the selected provider along with the question. Model support is required.
Each upload is limited to 4 MiB. Unattached uploads are retained in the local database;
referenced attachments are removed when their last conversation is deleted.

## Add knowledge

Open Knowledge base and upload a UTF-8 text file, Markdown, CSV, JSON, source code,
or a text-based PDF. `examples/project-notes.md` is fictional sample data you can
upload to try questions such as:

- "When is the Nimbus beta scheduled? Cite the notes."
- "What are the beta acceptance criteria?"
- "What does the project currently do with social media links?"

PDF extraction is optional:

```bash
python3 -m pip install -r requirements-optional.txt
```

Scanned PDFs need OCR, which this project does not implement. Limits: 4 MiB per
upload, 200 pages per PDF, 400,000 extracted characters per document, and 5,000
chunks per workspace.

The default search is lexical BM25 and needs no model call. For semantic search:

```dotenv
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

Or use local embeddings:

```bash
ollama pull embeddinggemma
```

```dotenv
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=embeddinggemma
```

Restart Atlas and upload documents after enabling embeddings. Existing documents
are not automatically re-embedded. When changing embedding models, delete and
re-upload documents you want searched semantically; vectors from different models
are never mixed. Embedding calls can send the complete document's chunks to the
configured embedding provider, not just the passages used in a chat.

Hybrid search combines lexical and cosine-similarity rankings. If a semantic query
fails, keyword retrieval continues with a warning. Retrieved passage chips let you
inspect the actual text; they identify supplied evidence, not proof that the model's
answer is correct. Source labels are scoped to each answer. Follow-up questions use
the previous user question as an additional retrieval hint.

## Architecture and reading order

1. `settings.py`: environment configuration and browser-safe settings.
2. `store.py`: SQLite schema, transactions, conversations, documents, and memory.
3. `retrieval.py`: extraction, overlapping chunks, BM25, cosine search, and rank fusion.
4. `providers.py`: raw HTTP adapters, SSE/NDJSON parsing, image inputs, embeddings,
   URL citations, and artifact downloads.
5. `engine.py`: context assembly, cancellation, conversation locking, and persistence.
6. `server.py`: localhost API, validation, streaming, uploads, and static assets.
7. `static/app.js`: client state, incremental streams, Markdown, sources, and controls.

For each turn, the server:

1. Validates settings and reserves the conversation to prevent overlapping turns.
2. Persists the user message and a streaming assistant placeholder.
3. Retrieves relevant passages and loads explicitly saved memory.
4. Selects recent complete turns that fit an approximate character budget.
5. Calls the chosen model and forwards public answer text/status events.
6. Saves partial progress, then final status, usage, citations, and artifact metadata.

The OpenAI provider's built-in tools handle their own execution loop. Atlas does
not expose arbitrary local shell execution. It does not display private model
reasoning tokens. Reasoning effort is a model control, not a request to reveal
chain-of-thought. Temperature is only sent to Ollama to avoid incompatibilities
with hosted reasoning models.

The context budget is character-based, not an exact tokenizer measurement. Images
also consume provider context. Large prompts may still exceed a model's limits.
Old turns are pruned as complete pairs and the newest question is retained. There
is no automatic summarization or hidden persistent model state between requests.

Cancellation stops at the next provider event; an idle provider may keep the UI
waiting until the upstream read timeout (120 seconds). Requests are not retried
automatically after streaming starts, avoiding accidental duplicate model charges.

## Testing

Core tests need no extra packages:

```bash
python3 -m unittest discover -s tests -v
node --test tests/frontend.cjs
```

The Python suite uses local mock model endpoints to verify real HTTP wire formats,
stream parsing, and application persistence. It does not measure model intelligence
or establish that hosted models/tools are available to your account.

An optional browser test requires Playwright and Chromium:

```bash
npm install --no-save playwright
npx playwright install chromium
```

Run an isolated demo server in one terminal:

```bash
ATLAS_DATA_DIR=/tmp/atlas-browser-test python3 app.py --demo --port 8765
```

Then run:

```bash
node tests/browser.cjs
```

This test adds a document, memory, and conversation to its workspace. It verifies
UI interactions and creates preview screenshots. Use a fresh temporary workspace
for each run. `ATLAS_TEST_URL` can select a different local address.

## Optional live evaluation

After connecting a real model, run:

```bash
python3 evaluate.py --provider ollama
# Or, with OPENAI_API_KEY configured:
python3 evaluate.py --provider openai
```

This makes real model requests, which may incur charges. It uses an isolated
temporary database, adds the fictional project notes and a memory, and checks four
small cases: two document facts, a saved preference, and conversation recall. Its
exact-string checks are intentionally simple and may reject valid paraphrases.
They are smoke tests, not a general quality benchmark. Full answers and metadata
are written to `evaluation-results.json` for review.

## Workspace and operating limits

Chats, extracted document text, embeddings, memory, and image bytes are stored in
`data/atlas.sqlite3` by default. The database is not encrypted. Browser preferences
are stored in localStorage. `.env` supplies server secrets and is excluded by the
included `.gitignore`. Environment variables take precedence over `.env` values.

The app binds to `127.0.0.1` and is intended for one user's local machine. Host and
Origin checks limit cross-origin access; generated HTML is escaped and the UI uses
a Content Security Policy. These controls do not make prompt injection impossible.
Do not expose this server publicly without adding authentication, user isolation,
rate limits, HTTPS, and an appropriate production server.

Run only one Atlas process against a workspace. A restart marks streaming messages
as interrupted. To back up the SQLite files, stop the server first and copy the
data directory. Exporting a conversation creates Markdown text, not a database
backup or a copy of its image attachments. Deleting a conversation removes local
records; it does not delete data already transmitted to an external provider.

No deployment, model download, account connection, or live inference is bundled
into the archive. You supply an API key or a running local model. See
`VERIFICATION.md` for the checks actually performed during creation.

## Official references used

- OpenAI model documentation: https://developers.openai.com/api/docs
- Streaming: https://developers.openai.com/api/docs/guides/streaming-responses
- Web search: https://developers.openai.com/api/docs/guides/tools-web-search
- Code interpreter: https://developers.openai.com/api/docs/guides/tools-code-interpreter
- Image input: https://developers.openai.com/api/docs/guides/images-vision
- Embeddings: https://developers.openai.com/api/docs/guides/embeddings
- Container file download: https://developers.openai.com/api/reference/resources/containers/subresources/files/subresources/content/methods/retrieve
- Ollama chat: https://docs.ollama.com/api/chat
- Ollama embeddings: https://docs.ollama.com/api/embed
