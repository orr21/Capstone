# Research Copilot — Architecture Blueprint & Data Model

## Diagrama de Arquitectura

```
                       ┌─────────────────────────────────┐
                       │   OpenAlex REST API (Inverted)  │
                       └────────────────┬────────────────┘
                                        │ Batch HTTP Ingestion
                                        ▼
                       ┌─────────────────────────────────┐
                       │   Ingestion & Enrichment        │
                       │  - Inverted Index Reconstruction│
                       │  - Postgres/Lakebase Upserts    │
                       └────────────────┬────────────────┘
                                        │
                                        ▼
                       ┌─────────────────────────────────┐
                       │ Databricks Lakebase Postgres DB │
                       │ (users, papers, conversations,  │
                       │  reading_plans, notes, etc.)    │
                       └────────┬────────────────┬───────┘
                                │                │
              Vector Embeddings │                │ SQL Read/Write Tools
              (SentenceTransf.) ▼                ▼
              ┌────────────────────┐   ┌─────────────────────────┐
              │  Paper Embeddings  │   │  FastMCP Tools Server   │
              │  Vector Table      │◄──┤  (mcp-research-copilot) │
              └────────────────────┘   └────────────┬────────────┘
                                                    │ FastMCP Client / Bearer Auth
                                                    ▼
                                       ┌─────────────────────────┐
                                       │   FastAPI UI Backend    │
                                       │   (Databricks App UI)   │
                                       └────────────┬────────────┘
                                                    │ Interactive SSE Stream
                                                    ▼
                                       ┌─────────────────────────┐
                                       │  ElevenLabs Editorial   │
                                       │     Frontend Chat UI    │
                                       └─────────────────────────┘
```

---

## 1. Lakehouse Data Model (Databricks Lakebase & Delta Schema)

Toda la persistencia relacional y vectorial se gestiona en la base de datos **Databricks Lakebase (PostgreSQL / Delta Lake)**.

| Table Name | Key Schema / Primary Key | Purpose |
|---|---|---|
| `users` | `user_id` (PK), `name`, `email`, `created_at` | Perfiles de cuenta de usuario. |
| `conversations` | `conversation_id` (PK), `user_id` (FK), `title`, `created_at`, `updated_at` | Sesiones de conversación del usuario con el Copilot. |
| `messages` | `message_id` (PK), `conversation_id` (FK), `role` (`user`, `assistant`, `tool`), `content`, `citations` (JSONB), `created_at` | Historial de mensajes y citas por conversación. |
| `learning_goals` | `goal_id` (PK), `user_id` (FK), `title`, `description`, `target_date`, `created_at` | Objetivos de aprendizaje definidos por el estudiante. |
| `papers` | `paper_id` (PK), `doi`, `title`, `abstract_text`, `publication_year`, `citation_count`, `open_access_url`, `topics` (JSONB) | Catálogo de papers ingestados desde OpenAlex. |
| `authors` | `author_id` (PK), `display_name`, `orcid`, `institution_name` | Metadatos de autores de investigación. |
| `paper_authors` | `(paper_id, author_id)` | Tabla junction que mapea papers a autores con orden de autoría. |
| `collections` | `collection_id` (PK), `user_id` (FK), `name`, `description`, `created_at` | Carpetas para guardar listas curadas de papers. |
| `collection_papers` | `(collection_id, paper_id)` | Papers asociados a colecciones específicas. |
| `reading_progress` | `progress_id` (PK), `user_id` (FK), `paper_id` (FK), `status` (`TO_READ`, `READING`, `COMPLETED`), `updated_at` | Seguimiento de lectura por paper por usuario. |
| `reading_plans` | `plan_id` (PK), `user_id` (FK), `goal_id` (FK), `title`, `sequenced_paper_ids` (JSONB), `status`, `created_at` | Planes de lectura estructurados y secuenciados por la IA. |
| `notes` | `note_id` (PK), `user_id` (FK), `paper_id` (FK), `content`, `created_at` | Notas y síntesis personales del estudiante sobre papers. |
| `paper_embeddings` | `paper_id` (PK/FK), `embedding` (`vector(384)`) | Embeddings semánticos para la búsqueda vectorial. |

---

## 2. Ingestion & Unstructured Data Pipeline

### Flujo de Ingestion

1. **API Fetching:** Job Python que recupera registros de papers desde la API de OpenAlex, filtrando por topics y keywords asociados a los `learning_goals` activos.
2. **Abstract Reconstruction:** OpenAlex provee los abstracts como un diccionario de índice invertido (`{"word": [positions]}`). Se ejecuta una reconstrucción limpia del texto:

```python
def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""
    word_position_pairs = [
        (pos, word)
        for word, positions in inverted_index.items()
        for pos in positions
    ]
    return " ".join([word for _, word in sorted(word_position_pairs)])
```

3. **Idempotent Upserts:** Uso de queries `ON CONFLICT (paper_id) DO UPDATE` para garantizar que la ingesta sea 100% idempotente.

---

## 3. Context Engineering & Semantic Search

### Multi-Entity Evidence Retrieval

La capa de retrieval no envía colecciones enteras al LLM. En su lugar, realiza un **recuperación semántica cruzada** combinando:
1. **Abstracts y metadatos de papers:** Embeddings vectoriales cosine similarity sobre `abstract_text`.
2. **Notas del estudiante:** Contenido cargado desde la tabla `notes`.
3. **Objetivos de aprendizaje (`learning_goals`):** Contexto activo del usuario.

### Formato de Citas Estructuradas

Todos los retrieval tools formatean los resultados de papers con citas estandarizadas:
`• [paper_id] "Title" (Year) | Citations: N | Match Score: X.XX`

El agente utiliza estas llaves `[paper_id]` para incluir citas explícitas en sus respuestas finales.

---

## 4. AI Agent Capabilities & FastMCP Tool Suite

El agente interactúa con el Lakehouse mediante herramientas FastMCP desplegadas en `mcp-research-copilot-tools`:

### Retrieval & Learning Tools

- **`search_research_papers(query: str, top_k: int = 5)`** — Búsqueda semántica vectorial contra `paper_embeddings` y `papers`.
- **`find_papers_for_goal(user_id: str, goal_title_or_id: str)`** — Recupera papers alineados con un objetivo de aprendizaje del usuario.
- **`summarize_and_compare_papers(paper_ids: list[str])`** — Recupera metadatos, abstracts y notas de 2 a 5 papers para síntesis comparativa.
- **`track_progress_and_recommend(user_id: str, goal_id: str = "")`** — Analiza el progreso (`COMPLETED`, `READING`, `TO_READ`) y recomienda el siguiente paper.

### Action, Planning & Session Tools

- **`generate_sequenced_reading_plan(user_id: str, goal_id: str, plan_title: str, paper_ids: list[str])`** — Crea y guarda un plan de lectura ordenado en `reading_plans`.
- **`add_paper_to_collection(user_id: str, collection_name: str, paper_id: str)`** — Añade o asocia un paper a una colección del usuario.
- **`save_paper_note(user_id: str, paper_id: str, note_content: str)`** — Guarda o actualiza notas de estudio personales.
- **`update_reading_progress(user_id: str, paper_id: str, status: str)`** — Actualiza el estado (`TO_READ`, `READING`, `COMPLETED`).
- **`manage_conversation_history(user_id: str, action: str, ...)`** — Registra sesiones de chat, conversaciones y mensajes con citas asociadas.

---

## 5. Databricks App Frontend (FastAPI + ElevenLabs Editorial UI)

Hospedado como una Databricks App nativa:

- **Editorial UI Design:** Estética inspirada en *ElevenLabs* (fondo off-white `#f5f5f5`, tipografía EB Garamond para títulos, Inter para cuerpo, micro-animaciones orb y tarjetas suaves).
- **Copilot Chat Interface:** Chat con streaming SSE (`/api/chat/stream`) que invoca al agente LLM (`databricks-meta-llama-3-3-70b-instruct`) con tool-calling dinámico a FastMCP.
- **MLflow Tracing:** Seguimiento de trazas de GenAI registrado en MLflow Experiments mediante `mlflow.start_run()` y `mlflow.start_span()`.