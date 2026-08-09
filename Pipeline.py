# Databricks notebook source
# MAGIC %pip install -q sentence-transformers
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, current_timestamp
from pyspark.sql.types import StringType

spark = SparkSession.builder.getOrCreate()

def fetch_openalex_papers_by_date(
    topic: str, 
    start_date: str, 
    end_date: str, 
    max_results: int = 100
) -> list[dict]:
    """
    Fetch papers from OpenAlex filtered by topic and publication date range.
    Example dates: start_date='2025-01-01', end_date='2026-08-01'
    """
    # Build OpenAlex filter query
    filter_clause = f"default.search:{topic},from_publication_date:{start_date},to_publication_date:{end_date}"
    url = f"https://api.openalex.org/works?filter={filter_clause}&per-page={max_results}&sort=publication_date:desc"
    
    headers = {"User-Agent": "mailto:orr21.personal@gmail.com"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        print(f"API Request Failed: {response.status_code} - {response.text}")
        return []

# Example: Fetch papers published between Jan 2025 and August 2026
raw_papers = fetch_openalex_papers_by_date(
    topic="data engineering", 
    start_date="2025-01-01", 
    end_date="2026-08-01", 
    max_results=50
)

# COMMAND ----------

import psycopg2
from psycopg2.extras import execute_values

# 1. Helper function to reconstruct abstract text
def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""
    word_pos = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_pos.append((pos, word))
    word_pos.sort(key=lambda x: x[0])
    return " ".join([word for _, word in word_pos])

# 2. Flatten raw JSON records into a list of TUPLES
parsed_records = []
for p in raw_papers:
    parsed_records.append((
        str(p.get("id")),
        p.get("doi"),
        p.get("title") or "",
        reconstruct_abstract(p.get("abstract_inverted_index")),
        p.get("publication_year"),
        p.get("cited_by_count", 0),
        (p.get("open_access") or {}).get("oa_url")
    ))

# 3. Connection params
DB_HOST = os.environ.get("DB_HOST", "ep-withered-breeze-d8845p1k.database.us-east-2.cloud.databricks.com")
DB_NAME = os.environ.get("DB_NAME", "databricks_postgres")
DB_USER = os.environ.get("DB_USER", "research-copilot-agent")
DB_PASS = os.environ.get("DB_PASS", "REDACTED")

conn = psycopg2.connect(
    host=DB_HOST,
    port=5432,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS,
    sslmode="require"
)

try:
    cursor = conn.cursor()
    
    insert_sql = """
        INSERT INTO papers (
            paper_id, doi, title, abstract_text, publication_year, citation_count, open_access_url
        ) VALUES %s
        ON CONFLICT (paper_id) DO UPDATE SET
            doi = EXCLUDED.doi,
            title = EXCLUDED.title,
            abstract_text = EXCLUDED.abstract_text,
            publication_year = EXCLUDED.publication_year,
            citation_count = EXCLUDED.citation_count,
            open_access_url = EXCLUDED.open_access_url;
    """
    
    execute_values(cursor, insert_sql, parsed_records, page_size=100)
    conn.commit()
    print(f"✅ Successfully inserted {len(parsed_records)} papers into Lakebase!")

finally:
    cursor.close()
    conn.close()

# COMMAND ----------

import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

# Active Lakebase credentials
DB_HOST = os.environ.get("DB_HOST", "ep-withered-breeze-d8845p1k.database.us-east-2.cloud.databricks.com")
DB_NAME = os.environ.get("DB_NAME", "databricks_postgres")
DB_USER = os.environ.get("DB_USER", "research-copilot-agent")
DB_PASS = os.environ.get("DB_PASS", "REDACTED")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 1. Load local embedding model
print(f"Loading embedding model '{MODEL_NAME}'...")
model = SentenceTransformer(MODEL_NAME)

# 2. Build text payload for embedding (Title + Abstract)
# parsed_records format from previous step: (paper_id, doi, title, abstract_text, year, citations, url)
texts_to_embed = [
    f"{r[2]}. {r[3]}" if r[3] else r[2] 
    for r in parsed_records
]

print(f"Computing embeddings for {len(texts_to_embed)} papers...")
embeddings = model.encode(texts_to_embed, show_progress_bar=True)

# 3. Format vector payload for pgvector
vector_records = [
    (
        r[0],  # paper_id
        r[2],  # title
        r[3][:300] if r[3] else "",  # abstract_snippet
        '{' + ','.join(str(float(x)) for x in vec) + '}',  # Postgres array literal
        MODEL_NAME
    )
    for r, vec in zip(parsed_records, embeddings)
]

# 4. Upsert into paper_embeddings
conn = psycopg2.connect(
    host=DB_HOST, port=5432, dbname=DB_NAME,
    user=DB_USER, password=DB_PASS, sslmode="require"
)

try:
    cursor = conn.cursor()
    
    insert_vector_sql = """
        INSERT INTO paper_embeddings (
            paper_id, title, abstract_snippet, embedding, model_name
        ) VALUES %s
        ON CONFLICT (paper_id) DO UPDATE SET
            title = EXCLUDED.title,
            abstract_snippet = EXCLUDED.abstract_snippet,
            embedding = EXCLUDED.embedding,
            model_name = EXCLUDED.model_name,
            embedded_at = CURRENT_TIMESTAMP;
    """
    
    template = "(%s, %s, %s, %s::double precision[]::vector, %s)"
    execute_values(cursor, insert_vector_sql, vector_records, template=template, page_size=100)
    conn.commit()
    print(f"✅ Successfully written {len(vector_records)} vector embeddings to Lakebase!")

finally:
    cursor.close()
    conn.close()