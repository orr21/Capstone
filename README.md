# Research Copilot

Semantic research paper search, note-taking, and reading-progress tracking powered by Databricks Lakebase, pgvector, and LLMs. Ask questions in natural language and the agent searches papers, retrieves metrics, saves notes, and updates your reading progress — all from a single chat interface.

> **Architecture screenshot placeholder**
>
> ![Architecture diagram](docs/images/architecture.png)
>
> *Replace with a diagram of the end-to-end flow: OpenAlex → Pipeline → Lakebase → MCP server → UI agent.*

---

## Features

- **Semantic paper search** — vector-similarity search over paper titles + abstracts using `all-MiniLM-L6-v2` embeddings stored in pgvector.
- **Research paper ingestion** — a Databricks notebook pulls papers from the [OpenAlex](https://openalex.org) API, upserts them into Lakebase, and computes embeddings.
- **Natural-language agent** — a chat UI backed by an LLM (Databricks serving endpoint) that discovers MCP tools and calls them to answer questions.
- **Notes & reading progress** — save per-paper notes and track `TO_READ` / `READING` / `COMPLETED` statuses.
- **Full observability** — every agent run is traced end-to-end with MLflow GenAI tracing (AGENT / TOOL / CHAT_MODEL spans).

> **UI screenshot placeholder**
>
> ![Research Copilot chat UI](docs/images/ui-chat.png)
>
> *Replace with a screenshot of the chat interface showing a question, tool-call cards, and the streamed answer.*

> **Semantic search demo placeholder**
>
> ![Semantic search results](docs/images/ui-search-demo.png)
>
> *Replace with a screenshot of the agent answering a search query with cited papers and match scores.*

---

## Architecture

| Component | Purpose | Location |
|---|---|---|
| `Pipeline.py` | Databricks notebook: OpenAlex ingestion + embedding generation | root |
| `mcp_server/` | MCP tool server exposing search / count / notes / progress tools | `mcp_server/` |
| `research-copilot-ui/` | FastAPI chat backend + single-page frontend | `research-copilot-ui/` |
| **Lakebase (Postgres + pgvector)** | `papers`, `paper_embeddings`, `notes`, `reading_progress` tables | Databricks |

### Data flow

1. **Ingest** — `Pipeline.py` queries OpenAlex for papers in a topic + date range, reconstructs abstracts, and upserts them into the `papers` table.
2. **Embed** — the same notebook encodes titles + abstracts with a local `sentence-transformers` model and upserts vectors into `paper_embeddings`.
3. **Serve tools** — the MCP server (`mcp_server/server.py`) exposes semantic search, counts, note-saving, and reading-progress updates against Lakebase.
4. **Chat** — the UI backend (`research-copilot-ui/server.py`) discovers MCP tools, streams them to an LLM over SSE, executes tool calls, and streams the synthesized answer back to the browser.

> **Data flow diagram placeholder**
>
> ![Pipeline data flow](docs/images/data-flow.png)
>
> *Replace with a diagram showing OpenAlex → papers → embeddings → vector search → agent → UI.*

### Database tables

| Table | Columns (key) | Purpose |
|---|---|---|
| `papers` | `paper_id`, `doi`, `title`, `abstract_text`, `publication_year`, `citation_count`, `open_access_url` | Paper metadata |
| `paper_embeddings` | `paper_id`, `title`, `abstract_snippet`, `embedding` (vector), `model_name` | Semantic vectors for search |
| `notes` | `note_id`, `user_id`, `paper_id`, `content` | Personal notes |
| `reading_progress` | `progress_id`, `user_id`, `paper_id`, `status`, `updated_at` | Reading status |

---

## Repository layout

```
Capstone/
├── Pipeline.py                  # Databricks notebook: OpenAlex ingestion + embeddings
├── ELEVENLABS-DESIGN.md         # Design system spec used by the UI
├── mcp_server/
│   ├── server.py                # FastMCP server (search / count / notes / progress)
│   ├── requirements.txt
│   └── app.yaml                 # Databricks App manifest
└── research-copilot-ui/
    ├── server.py                # FastAPI chat backend + agent orchestration
    ├── static/index.html        # Single-page chat UI
    ├── requirements.txt
    └── app.yaml                 # Databricks App manifest
```

---

## Getting started

### Prerequisites

- A Databricks workspace with a Lakebase Postgres database (pgvector enabled)
- A Databricks serving endpoint exposing an LLM (e.g. `databricks-meta-llama-3-3-70b-instruct`)
- Python 3.10+

### 1. Ingest papers into Lakebase

Run `Pipeline.py` as a Databricks notebook. Update the `topic`, `start_date`, and `end_date` arguments:

```python
raw_papers = fetch_openalex_papers_by_date(
    topic="data engineering",
    start_date="2025-01-01",
    end_date="2026-08-01",
    max_results=50
)
```

The notebook upserts paper metadata into `papers` and embeddings into `paper_embeddings`.

### 2. Deploy the MCP server

Deploy `mcp_server/` as a Databricks App (see `app.yaml`). Expose the following environment variables instead of hardcoding credentials:

```
DB_HOST
DB_NAME
DB_USER
DB_PASS
MODEL_NAME      # default: sentence-transformers/all-MiniLM-L6-v2
```

Run locally:

```bash
pip install -r mcp_server/requirements.txt
python mcp_server/server.py   # serves on :8000 at /mcp
```

### 3. Deploy the UI

Deploy `research-copilot-ui/` as a Databricks App (see `app.yaml`). Key environment variables:

```
DB_HOST
DB_NAME
DB_USER
DB_PASS
AGENT_ENDPOINT_NAME         # LLM serving endpoint
MCP_SERVERS_CONFIG          # JSON map of namespace -> MCP server URL
MLFLOW_EXPERIMENT_ID        # MLflow tracing experiment
```

Run locally:

```bash
pip install -r research-copilot-ui/requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000`.

---

## How the agent works

The chat endpoint (`POST /api/chat/stream`) returns an SSE stream of events:

| Event | Meaning |
|---|---|
| `status` | Progress updates ("Discovering MCP tools…", "Querying model…") |
| `tool_start` / `tool_end` | A tool was invoked and finished (rendered as expandable cards) |
| `chunk` | Streamed LLM answer text |
| `error` | Failure with a hint |

Flow: the backend loads all MCP tools, attaches them to an LLM call, executes any tool calls the model emits against the MCP server, then makes a final streaming call to synthesize the answer. Every run is wrapped in an MLflow `AGENT` trace with nested `TOOL` and `CHAT_MODEL` spans.

---

## MCP tools

| Tool | Description |
|---|---|
| `search_research_papers(query, top_k)` | Semantic vector search over papers, returning titles, years, citation counts, and match scores |
| `count_papers(query, publication_year)` | Count papers, optionally filtered by keyword or year |
| `save_paper_note(user_id, paper_id, note_content)` | Upsert a personal note for a paper |
| `update_reading_progress(user_id, paper_id, status)` | Set `TO_READ`, `READING`, or `COMPLETED` |

> **MCP tool output placeholder**
>
> ![MCP tool card](docs/images/ui-tool-card.png)
>
> *Replace with a screenshot of the tool-call cards in the chat UI showing a running/completed search with viewable output.*

---

## Observability

MLflow GenAI tracing records each agent conversation in the configured experiment (`/Shared/research-copilot-tracing` by default). Each trace captures:

- The user's question
- Every tool call with its inputs and outputs
- The LLM model and emitted tool calls
- The final answer

> **MLflow trace placeholder**
>
> ![MLflow trace](docs/images/mlflow-trace.png)
>
> *Replace with a screenshot of an MLflow trace: agent span → tool spans → chat model spans.*

---

## Security notes

- **Do not commit credentials.** The repo currently contains database and API credentials in `Pipeline.py` and both `app.yaml` files. Replace these with Databricks secrets or environment variables before sharing or deploying to production.
- The MCP server app must be reachable from the UI app. If the MCP app has access control enabled, set it to "Open access" or configure M2M auth for the UI.
- Database connections use SSL (`sslmode=require`).

---

## Roadmap ideas

- [ ] Multi-user notes and shared reading lists
- [ ] Filter by venue, author, or citation threshold
- [ ] Reranking / hybrid keyword + vector search
- [ ] Export search results (BibTeX, CSV)
