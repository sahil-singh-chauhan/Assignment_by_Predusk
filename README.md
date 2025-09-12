# Assignment_by_Predusk
It is a mini rag built as an assignment.

# PDF RAG Chat (Flask)

Upload a PDF and ask grounded questions about it. The app uses a RAG pipeline: embeddings → vector DB retrieval → rerank → LLM answer with inline citations and real source snippets.

## 1) Vector Database (hosted)
- Provider: Pinecone (cloud-hosted)
- Index/collection: `PINECONE_INDEX_NAME` (env, default `apirag`)
- Dimensionality: auto-detected from the embedding model at startup (current model = 384 dims)
- Upsert strategy:
  - PDF is split into chunks; each chunk is embedded and upserted with a unique id `chunk_{i}_{namespace}`
  - Per-session namespace (e.g., `pdf_xxxxxxxx`) isolates data per upload
  - Metadata stored per vector: `source` (path), `title` (filename), `section` (page if available), `position` (chunk index)

## 2) Embeddings & Chunking
- Embedding model: Sentence Transformers (`EMBEDDING_MODEL` env; default `latterworks/ollama-embeddings` → 384-d)
  - You can switch to any provider (OpenAI/Cohere/Jina/Voyage/Nomic) by changing env and wrapper
- Chunking strategy: token-based splitter `from_tiktoken_encoder`
  - Size: 1,000 tokens
  - Overlap: 150 tokens (~15%)
- Metadata captured on each chunk for citations: `source`, `title`, `section` (page), `position`

## 3) Retriever + Reranker
- Retriever: custom Pinecone retriever (top-k=5) + LangChain `MultiQueryRetriever` to expand queries
- Reranker: `PineconeRerank` with `top_n=5` applied before answering
  - You can replace with Cohere/Jina/Voyage/BGE rerankers by swapping the reranker component

## 4) LLM & Answering
- LLM provider: `ChatOpenAI` via `OPENAI_API_BASE` and `OPENAI_API_KEY`
  - Works with OpenRouter (default) or OpenAI (set `OPENAI_API_BASE=https://api.openai.com/v1`)
  - Model id: `OPENAI_MODEL`
- Grounding & citations:
  - The prompt instructs the model to answer using only retrieved context
  - Inline citations like `[1]`, `[2]` map to numbered context items
  - Backend appends real source snippets corresponding only to the citations used in the answer
- No-answer handling: if context lacks relevant info, model is instructed to respond accordingly

## 5) Frontend
- UI: upload area, chat box, answers panel with citations & sources
- UX:
  - Uploading popup: “Uploading PDF… this may take up to 2 minutes” (shown during indexing)
  - Success toast when indexing completes
- Metrics:
  - UI shows response time (ms)
  - Backend returns rough token and cost estimates in the `/chat` response JSON (not shown in UI by default)

## Setup
1) Create venv and install deps
```bash
py -m venv .venv
# PowerShell
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
2) Create `.env`
```bash
OPENAI_API_KEY=your_key
OPENAI_API_BASE=https://openrouter.ai/api/v1  # or https://api.openai.com/v1
OPENAI_MODEL=openai/gpt-4o-mini
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=apirag
EMBEDDING_MODEL=latterworks/ollama-embeddings
FLASK_SECRET_KEY=change-me
```
3) Run
```bash
python app.py
```
Open http://localhost:5000

## Notes
- OpenRouter Zero Data Retention: use a ZDR-supported model or disable ZDR in settings
- Index is auto-created with the detected embedding dimension
- Each page refresh clears prior session’s namespace

## Troubleshooting
- 0 documents retrieved: re-upload PDF and ask without refreshing between steps
- 404 from OpenRouter (ZDR): select a ZDR-capable model or disable ZDR
- Slow upload: large files and embedding can take up to ~2 minutes

---

## Architecture & Settings (Compact)
- Vector DB: Pinecone (cloud). Index: `PINECONE_INDEX_NAME` (default `apirag`). Dim: inferred from embeddings (384). Upsert: per-chunk ids `chunk_{i}_{namespace}`; namespace per session.
- Embeddings: Sentence Transformers (`EMBEDDING_MODEL`, default `latterworks/ollama-embeddings`, 384-d). Chunking: 1,000 tokens, 150 overlap (~15%). Metadata per chunk: `source`, `title`, `section`, `position`.
- Retriever: custom Pinecone retriever (top-k 5) + MultiQuery retriever. Reranker: PineconeRerank top_n 5.
- LLM: OpenRouter-compatible `ChatOpenAI` (answer-only). Grounded answer with inline citations; backend appends only cited snippets. No-answer handled in prompt.
- Frontend: upload area + chat box + answers panel; shows time; tokens and rough cost appended under the answer.
- Hosting: Render (Gunicorn). Keep keys server-side. Example env in `env.example`.

## Remarks
- OpenRouter ZDR may block some models; choose a ZDR-supported model or disable ZDR.
- Render free tier is resource-limited; service runs with 1 worker / 2 threads to reduce memory.
