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
DB_PASS  = os.environ.get("DB_PASS", "npg_oG8qxChTtIz3")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Ingestion Parameters
# ---------------------------------------------------------------------------
TOPICS = []

# Filter only freely accessible (Open Access) papers online
ONLY_OPEN_ACCESS = True

# Filter strictly for English language publications
ONLY_ENGLISH = True

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

BATCH_WRITE_SIZE = int(os.environ.get("INGEST_BATCH_SIZE", "1000"))

# ---------------------------------------------------------------------------
# Helpers & Database Checkpointing
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


def get_existing_paper_ids() -> set:
    """Fetch existing paper IDs from Lakebase to resume ingestion without duplicate work."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT paper_id FROM papers")
            rows = cur.fetchall()
            return {r[0] for r in rows}
    except Exception as e:
        print(f"  Warning fetching existing papers: {e}")
        return set()
    finally:
        conn.close()


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


def build_filter_clause(topic, start_date: str, end_date: str, only_oa: bool = True, only_english: bool = True) -> str:
    """Build OpenAlex filter query string."""
    filters = [
        f"from_publication_date:{start_date}",
        f"to_publication_date:{end_date}"
    ]
    if only_oa:
        filters.append("is_oa:true")
    if only_english:
        filters.append("language:en")
    if topic:
        filters.append(f"default.search:{topic}")

    return ",".join(filters)


def fetch_openalex_page(filter_clause: str, cursor: str = "*", per_page: int = 200) -> dict:
    """OpenAlex Bulk REST API Fetching Optimization."""
    select_fields = "id,doi,title,abstract_inverted_index,publication_year,cited_by_count,open_access,language,authorships"
    
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


def parse_papers(raw_papers: list, topic_label=None) -> list:
    """Flatten raw OpenAlex JSON into tuples for PostgreSQL `papers` table ingestion."""
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
        ))
    return records


def parse_authors(raw_papers: list) -> tuple:
    """Extract author tuples and paper-author junction tuples from raw OpenAlex records."""
    authors_dict = {}  # author_id -> (author_id, display_name, orcid, institution_name)
    paper_authors_list = []  # (paper_id, author_id, author_position)

    for p in raw_papers:
        paper_id = str(p.get("id"))
        authorships = p.get("authorships") or []
        for idx, a_item in enumerate(authorships):
            author_info = a_item.get("author") or {}
            raw_aid = author_info.get("id")
            if not raw_aid:
                continue
            author_id = str(raw_aid)
            display_name = (author_info.get("display_name") or "").strip()
            orcid = (author_info.get("orcid") or "").strip()
            
            institutions = a_item.get("institutions") or []
            institution_name = (institutions[0].get("display_name") or "").strip() if institutions else ""

            authors_dict[author_id] = (author_id, display_name, orcid, institution_name)
            paper_authors_list.append((paper_id, author_id, idx + 1))

    return list(authors_dict.values()), paper_authors_list


def ingest_papers(parsed_records: list):
    """Upsert paper metadata into PostgreSQL `papers` table."""
    if not parsed_records:
        return

    insert_sql = """
        INSERT INTO papers (
            paper_id, doi, title, abstract_text,
            publication_year, citation_count, open_access_url
        ) VALUES %s
        ON CONFLICT (paper_id) DO UPDATE SET
            doi              = EXCLUDED.doi,
            title            = EXCLUDED.title,
            abstract_text    = EXCLUDED.abstract_text,
            publication_year = EXCLUDED.publication_year,
            citation_count   = EXCLUDED.citation_count,
            open_access_url  = EXCLUDED.open_access_url;
    """

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            execute_values(cur, insert_sql, parsed_records, page_size=100)
        conn.commit()
    finally:
        conn.close()


def ingest_authors(authors_records: list, paper_authors_records: list):
    """Upsert authors and paper-author junction records into Lakebase."""
    if not authors_records:
        return

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            insert_authors_sql = """
                INSERT INTO authors (author_id, display_name, orcid, institution_name)
                VALUES %s
                ON CONFLICT (author_id) DO UPDATE SET
                    display_name     = EXCLUDED.display_name,
                    orcid            = EXCLUDED.orcid,
                    institution_name = EXCLUDED.institution_name;
            """
            execute_values(cur, insert_authors_sql, authors_records, page_size=100)

            if paper_authors_records:
                insert_pa_sql = """
                    INSERT INTO paper_authors (paper_id, author_id, author_position)
                    VALUES %s
                    ON CONFLICT (paper_id, author_id) DO UPDATE SET
                        author_position = EXCLUDED.author_position;
                """
                execute_values(cur, insert_pa_sql, paper_authors_records, page_size=100)

        conn.commit()
    except Exception as e:
        print(f"  Warning writing authors/paper_authors: {e}")
    finally:
        conn.close()


def embed_and_store(parsed_records: list, model: SentenceTransformer):
    """Generate embeddings and upsert into PostgreSQL `paper_embeddings` table."""
    if not parsed_records:
        return

    texts = [
        f"{r[2]}. {r[3]}" if r[3] else r[2]
        for r in parsed_records
    ]

    embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)

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
    finally:
        conn.close()


def process_batch(raw_batch: list, topic_label, embed_model: SentenceTransformer) -> int:
    """Parse papers and authors, ingest into PostgreSQL, compute embeddings, and commit immediately."""
    parsed_papers = parse_papers(raw_batch, topic_label)
    if not parsed_papers:
        return 0
    
    # 1. Ingest paper metadata
    ingest_papers(parsed_papers)

    # 2. Parse and ingest authors + paper_authors junction table
    authors_records, paper_authors_records = parse_authors(raw_batch)
    ingest_authors(authors_records, paper_authors_records)

    # 3. Compute vector embeddings
    embed_and_store(parsed_papers, embed_model)
    return len(parsed_papers)


# ---------------------------------------------------------------------------
# Streaming Ingestion Loop with Incremental Checkpointing
# ---------------------------------------------------------------------------

def run_incremental_pipeline():
    print("=" * 60)
    print("Daily Incremental Ingestion Pipeline (Streaming Checkpoints)")
    print(f"  Window      : {START_DATE} -> {END_DATE} (Last {LOOKBACK_DAYS} days)")
    print(f"  Open Access : {ONLY_OPEN_ACCESS}")
    print(f"  Batch Size  : {BATCH_WRITE_SIZE} papers / checkpoint")
    print(f"  Topics      : {TOPICS if TOPICS else 'ALL (no keyword restriction)'}")
    print(f"  Max total   : {MAX_TOTAL or 'unlimited'}")
    print("=" * 60)

    print("\nLoading existing paper IDs from Lakebase for resume capability...")
    existing_pids = get_existing_paper_ids()
    print(f"  Found {len(existing_pids)} papers already committed in Lakebase.")

    print(f"\nLoading embedding model '{MODEL_NAME}'...")
    embed_model = SentenceTransformer(MODEL_NAME)

    jobs = TOPICS if TOPICS else [None]
    total_processed_session = 0

    for topic in jobs:
        label = f'"{topic}"' if topic else "all topics"
        print(f"\nStreaming ingestion for {label}...")

        filter_clause = build_filter_clause(topic, START_DATE, END_DATE, only_oa=ONLY_OPEN_ACCESS, only_english=ONLY_ENGLISH)
        cursor = "*"
        page_count = 0
        current_batch = []

        while cursor:
            data = fetch_openalex_page(filter_clause, cursor=cursor, per_page=PER_PAGE)
            page_results = data.get("results", [])
            total_available = data.get("meta", {}).get("count", 0)
            next_cursor = data.get("meta", {}).get("next_cursor")

            if not page_results:
                break

            # Filter out papers already committed to Lakebase
            new_papers = [p for p in page_results if str(p.get("id")) not in existing_pids]
            current_batch.extend(new_papers)
            page_count += 1
            
            # Track newly seen paper IDs to avoid intra-run duplicates
            for p in new_papers:
                existing_pids.add(str(p.get("id")))

            print(f"  Page {page_count}: {len(page_results)} fetched, {len(new_papers)} new (Pending batch: {len(current_batch)} / {BATCH_WRITE_SIZE})")

            # Immediately ingest & embed when batch reaches threshold
            if len(current_batch) >= BATCH_WRITE_SIZE:
                num_written = process_batch(current_batch, topic, embed_model)
                total_processed_session += num_written
                print(f"  💾 CHECKPOINT COMMITTED: {num_written} papers ingested & embedded! (Session total: {total_processed_session})")
                current_batch = []

            if MAX_TOTAL and total_processed_session >= MAX_TOTAL:
                print(f"  Reached MAX_TOTAL limit ({MAX_TOTAL}). Stopping.")
                break

            if not next_cursor or next_cursor == cursor:
                break

            cursor = next_cursor
            time.sleep(REQUEST_DELAY)

        # Process any remaining papers in final batch for this topic
        if current_batch:
            num_written = process_batch(current_batch, topic, embed_model)
            total_processed_session += num_written
            print(f"  💾 FINAL TOPIC CHECKPOINT COMMITTED: {num_written} papers!")

    print(f"\nPipeline Execution Complete! Total new papers ingested & embedded: {total_processed_session}")


if __name__ == "__main__":
    run_incremental_pipeline()