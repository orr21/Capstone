# Databricks notebook source
# MAGIC %pip install -q sentence-transformers
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import time
import requests
import psycopg2
from datetime import date, timedelta
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_HOST  = os.environ.get("DB_HOST", "ep-withered-breeze-d8845p1k.database.us-east-2.cloud.databricks.com")
DB_NAME  = os.environ.get("DB_NAME", "databricks_postgres")
DB_USER  = os.environ.get("DB_USER", "research-copilot-agent")
DB_PASS  = os.environ.get("DB_PASS", "")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Ingestion Parameters
# ---------------------------------------------------------------------------
TOPICS = []

# Filter only freely accessible (Open Access) papers online
ONLY_OPEN_ACCESS = True

# Dynamic rolling date window: (today - LOOKBACK_DAYS) to today
LOOKBACK_DAYS = int(os.environ.get("INGEST_LOOKBACK_DAYS", "3"))

today = date.today()
default_start = (today - timedelta(days=LOOKBACK_DAYS)).isoformat()
default_end = today.isoformat()

START_DATE = os.environ.get("INGEST_START_DATE", default_start)
END_DATE   = os.environ.get("INGEST_END_DATE", default_end)

PER_PAGE = 200  # Max per page in OpenAlex API
MAX_TOTAL = int(os.environ.get("INGEST_MAX_TOTAL", "1000")) if os.environ.get("INGEST_MAX_TOTAL") else None
REQUEST_DELAY = 0.1  # Polite pool allows up to 10 req/sec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_db_connection():
    """Return a new psycopg2 connection using env-configured credentials."""
    return psycopg2.connect(
        host=DB_HOST,
        port=5432,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        sslmode="require"
    )


def reconstruct_abstract(inverted_index: dict) -> str:
    """Rebuild plain-text abstract from OpenAlex inverted-index format."""
    if not inverted_index:
        return ""
    word_pos = [
        (pos, word)
        for word, positions in inverted_index.items()
        for pos in positions
    ]
    word_pos.sort()
    return " ".join(word for _, word in word_pos)


def build_filter_clause(topic, start_date: str, end_date: str, only_oa: bool = True) -> str:
    """
    Build OpenAlex filter query string.
    - `is_oa:true`: Filters strictly for freely available online Open Access papers.
    - `from_publication_date` / `to_publication_date`: Date range window.
    - `default.search`: Topic keyword (optional).
    """
    filters = [
        f"from_publication_date:{start_date}",
        f"to_publication_date:{end_date}"
    ]
    
    if only_oa:
        filters.append("is_oa:true")
        
    if topic:
        filters.append(f"default.search:{topic}")

    return ",".join(filters)


def fetch_openalex_page(filter_clause: str, cursor: str = "*", per_page: int = 200) -> dict:
    """
    OpenAlex Bulk REST API Fetching Optimization:
    1. `select=...`: Requests ONLY the fields we need. Shrinks response size by ~85%.
    2. `cursor=...`: Cursor pagination allows fetching millions of works without hitting page limits.
    3. `mailto=...`: Puts request into OpenAlex 'Polite Pool' for higher rate limits (10 req/s).
    """
    select_fields = "id,doi,title,abstract_inverted_index,publication_year,cited_by_count,open_access"
    
    url = (
        f"https://api.openalex.org/works"
        f"?filter={filter_clause}"
        f"&select={select_fields}"
        f"&per-page={per_page}"
        f"&cursor={cursor}"
        f"&sort=publication_date:desc"
        f"&mailto=orr21.personal@gmail.com"
    )
    headers = {"User-Agent": "ResearchCopilotAgent/1.0 (mailto:orr21.personal@gmail.com)"}

    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 5))
                print(f"  Rate-limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                break
        except requests.RequestException as e:
            print(f"  Request error (attempt {attempt+1}/3): {e}")
            time.sleep(2 ** attempt)

    return {}


def fetch_all_papers(topic, start_date: str, end_date: str, max_total=None) -> list:
    """Bulk-fetch papers from OpenAlex using cursor-based pagination."""
    label = f'"{topic}"' if topic else "all topics"
    print(f"\nFetching papers for {label} ({start_date} -> {end_date}, Open Access only={ONLY_OPEN_ACCESS})...")

    filter_clause = build_filter_clause(topic, start_date, end_date, only_oa=ONLY_OPEN_ACCESS)
    papers = []
    cursor = "*"
    page_count = 0

    while cursor:
        data = fetch_openalex_page(filter_clause, cursor=cursor, per_page=PER_PAGE)
        batch = data.get("results", [])
        total_available = data.get("meta", {}).get("count", 0)
        next_cursor = data.get("meta", {}).get("next_cursor")

        if not batch:
            break

        papers.extend(batch)
        page_count += 1
        print(f"  Batch {page_count}: fetched {len(batch)} (total: {len(papers)} / {total_available})")

        if max_total and len(papers) >= max_total:
            papers = papers[:max_total]
            print(f"  Reached MAX_TOTAL limit ({max_total}). Stopping.")
            break

        if not next_cursor or next_cursor == cursor:
            break

        cursor = next_cursor
        time.sleep(REQUEST_DELAY)

    print(f"  Collected {len(papers)} papers for {label}.")
    return papers


def parse_papers(raw_papers: list, topic_label) -> list:
    """Flatten raw OpenAlex JSON into tuples for PostgreSQL ingestion."""
    records = []
    for p in raw_papers:
        records.append((
            str(p.get("id")),
            p.get("doi"),
            p.get("title") or "",
            reconstruct_abstract(p.get("abstract_inverted_index")),
            p.get("publication_year"),
            p.get("cited_by_count", 0),
            (p.get("open_access") or {}).get("oa_url"),
            topic_label or "general",
        ))
    return records


# ---------------------------------------------------------------------------
# Ingestion Steps
# ---------------------------------------------------------------------------

def ingest_papers(parsed_records: list):
    """Upsert paper metadata into PostgreSQL `papers` table."""
    if not parsed_records:
        print("  No records to insert.")
        return

    insert_sql = """
        INSERT INTO papers (
            paper_id, doi, title, abstract_text,
            publication_year, citation_count, open_access_url, topic
        ) VALUES %s
        ON CONFLICT (paper_id) DO UPDATE SET
            doi              = EXCLUDED.doi,
            title            = EXCLUDED.title,
            abstract_text    = EXCLUDED.abstract_text,
            publication_year = EXCLUDED.publication_year,
            citation_count   = EXCLUDED.citation_count,
            open_access_url  = EXCLUDED.open_access_url,
            topic            = EXCLUDED.topic;
    """

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            execute_values(cur, insert_sql, parsed_records, page_size=100)
        conn.commit()
        print(f"  Upserted {len(parsed_records)} papers into `papers`.")
    finally:
        conn.close()


def embed_and_store(parsed_records: list, model: SentenceTransformer):
    """Generate embeddings and upsert into PostgreSQL `paper_embeddings` table."""
    if not parsed_records:
        print("  No records to embed.")
        return

    texts = [
        f"{r[2]}. {r[3]}" if r[3] else r[2]
        for r in parsed_records
    ]

    print(f"  Computing embeddings for {len(texts)} papers...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    vector_records = [
        (
            r[0],
            r[2],
            r[3][:300] if r[3] else "",
            "{" + ",".join(str(float(x)) for x in vec) + "}",
            MODEL_NAME,
        )
        for r, vec in zip(parsed_records, embeddings)
    ]

    insert_vector_sql = """
        INSERT INTO paper_embeddings (
            paper_id, title, abstract_snippet, embedding, model_name
        ) VALUES %s
        ON CONFLICT (paper_id) DO UPDATE SET
            title            = EXCLUDED.title,
            abstract_snippet = EXCLUDED.abstract_snippet,
            embedding        = EXCLUDED.embedding,
            model_name       = EXCLUDED.model_name,
            embedded_at      = CURRENT_TIMESTAMP;
    """
    template = "(%s, %s, %s, %s::double precision[]::vector, %s)"

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            execute_values(cur, insert_vector_sql, vector_records, template=template, page_size=100)
        conn.commit()
        print(f"  Upserted {len(vector_records)} embeddings into `paper_embeddings`.")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

print("=" * 60)
print("Daily Incremental Ingestion Pipeline")
print(f"  Window      : {START_DATE} -> {END_DATE} (Last {LOOKBACK_DAYS} days)")
print(f"  Open Access : {ONLY_OPEN_ACCESS}")
print(f"  Topics      : {TOPICS if TOPICS else 'ALL (no keyword restriction)'}")
print(f"  Max total   : {MAX_TOTAL or 'unlimited'}")
print("=" * 60)

print(f"\nLoading embedding model '{MODEL_NAME}'...")
embed_model = SentenceTransformer(MODEL_NAME)

jobs = TOPICS if TOPICS else [None]
all_parsed = []

for topic in jobs:
    raw = fetch_all_papers(topic, START_DATE, END_DATE, max_total=MAX_TOTAL)
    parsed = parse_papers(raw, topic_label=topic)

    ingest_papers(parsed)
    embed_and_store(parsed, embed_model)

    all_parsed.extend(parsed)

print(f"\nPipeline complete! Total papers processed: {len(all_parsed)}")
