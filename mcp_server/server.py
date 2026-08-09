import os
import sys
import logging
import pg8000.dbapi
from fastmcp import FastMCP
from starlette.responses import RedirectResponse

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("mcp-research-copilot")

# Initialize FastMCP Server
mcp = FastMCP("mcp-research-copilot-tools")

# Ensure root '/' redirects to '/mcp'
if hasattr(mcp, 'app') and mcp.app is not None:
    @mcp.app.get("/")
    async def redirect_to_mcp(request):
        return RedirectResponse(url="/mcp")

# Environment Variables
DB_HOST = os.environ.get("DB_HOST", "ep-withered-breeze-d8845p1k.database.us-east-2.cloud.databricks.com")
DB_NAME = os.environ.get("DB_NAME", "databricks_postgres")
DB_USER = os.environ.get("DB_USER", "research-copilot-agent")
DB_PASS = os.environ.get("DB_PASS", "REDACTED")
MODEL_NAME = os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(MODEL_NAME)
    return _embed_model

def get_db_conn():
    return pg8000.dbapi.connect(
        host=DB_HOST, port=5432, database=DB_NAME,
        user=DB_USER, password=DB_PASS, ssl_context=True
    )

@mcp.tool
def search_research_papers(query: str, top_k: int = 3) -> str:
    """
    Semantically searches research papers in Lakebase using vector similarity.
    """
    try:
        model = get_embed_model()
        q_vec = '{' + ','.join(str(float(x)) for x in model.encode([query])[0]) + '}'
        
        conn = get_db_conn()
        cursor = conn.cursor()
        
        sql = """
            SELECT 
                p.paper_id, p.title, p.abstract_text, p.publication_year, p.citation_count,
                1 - (pe.embedding <=> %s::double precision[]::vector) AS similarity
            FROM paper_embeddings pe
            JOIN papers p ON pe.paper_id = p.paper_id
            ORDER BY pe.embedding <=> %s::double precision[]::vector ASC
            LIMIT %s;
        """
        cursor.execute(sql, (q_vec, q_vec, top_k))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not rows:
            return "No relevant papers found in Lakebase."
            
        output = []
        for r in rows:
            snippet = r[2][:250] + "..." if r[2] else "No abstract available."
            output.append(
                f"• [{r[0]}] {r[1]} ({r[3]}) | Citations: {r[4]} | Match Score: {r[5]:.2f}\n  Abstract: {snippet}"
            )
        return "\n\n".join(output)
    except Exception as e:
        logger.exception("Search failed")
        return f"Error executing search: {str(e)}"

@mcp.tool
def count_papers(query: str = "", publication_year: int | None = None) -> dict:
    """
    Counts research papers in Lakebase, optionally filtered by a keyword in the
    title/abstract or by publication year. Returns the total number of matching papers.
    """
    where = []
    params = []
    if query:
        where.append("(title ILIKE %s OR abstract_text ILIKE %s)")
        params += [f"%{query}%", f"%{query}%"]
    if publication_year:
        where.append("publication_year = %s")
        params.append(publication_year)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM papers{where_sql}", params)
        total = cursor.fetchone()[0]
        return {"total_papers": total, "query": query, "publication_year": publication_year}
    except Exception as e:
        logger.exception("Count failed")
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()
        conn.close()

@mcp.tool
def save_paper_note(user_id: str, paper_id: str, note_content: str) -> dict:
    """
    Saves a personal research note or summary for a paper in Lakebase.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        note_id = f"note_{user_id}_{paper_id[:10]}"
        sql = """
            INSERT INTO notes (note_id, user_id, paper_id, content)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (note_id) DO UPDATE SET content = EXCLUDED.content;
        """
        cursor.execute(sql, (note_id, user_id, paper_id, note_content))
        conn.commit()
        return {"status": "success", "message": f"Saved note for paper {paper_id}"}
    finally:
        cursor.close()
        conn.close()

@mcp.tool
def update_reading_progress(user_id: str, paper_id: str, status: str) -> dict:
    """
    Updates reading progress for a paper ('TO_READ', 'READING', or 'COMPLETED').
    """
    status_clean = status.upper()
    if status_clean not in ['TO_READ', 'READING', 'COMPLETED']:
        return {"status": "error", "message": "Invalid status."}
        
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        progress_id = f"prog_{user_id}_{paper_id[:10]}"
        sql = """
            INSERT INTO reading_progress (progress_id, user_id, paper_id, status)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id, paper_id) DO UPDATE SET status = EXCLUDED.status, updated_at = CURRENT_TIMESTAMP;
        """
        cursor.execute(sql, (progress_id, user_id, paper_id, status_clean))
        conn.commit()
        return {"status": "success", "message": f"Updated status of {paper_id} to {status_clean}"}
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    port = int(os.getenv("DATABRICKS_APP_PORT", os.getenv("PORT", 8000)))
    mcp.run(transport="http", host="0.0.0.0", port=port)