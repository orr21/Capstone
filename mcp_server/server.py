import os
import sys
import json
import logging
import uuid
import pg8000.dbapi
from typing import List, Dict, Any, Optional
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
def search_research_papers(query: str, top_k: int = 5) -> str:
    """
    Semantically searches research papers in Lakebase using vector similarity.
    Returns paper IDs, titles, publication years, citations, and abstracts.
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
            snippet = r[2][:300] + "..." if r[2] else "No abstract available."
            output.append(
                f"• [{r[0]}] \"{r[1]}\" ({r[3]}) | Citations: {r[4]} | Match Score: {r[5]:.2f}\n  Abstract: {snippet}"
            )
        return "\n\n".join(output)
    except Exception as e:
        logger.exception("Search failed")
        return f"Error executing search: {str(e)}"

@mcp.tool
def find_papers_for_goal(user_id: str, goal_title_or_id: str) -> str:
    """
    Finds papers matching a student's learning goal by querying learning goals and searching relevant papers.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        # Find matching goal description or title
        cursor.execute(
            "SELECT goal_id, title, description FROM learning_goals WHERE user_id = %s AND (goal_id = %s OR title ILIKE %s) LIMIT 1",
            (user_id, goal_title_or_id, f"%{goal_title_or_id}%")
        )
        goal_row = cursor.fetchone()
        cursor.close()
        conn.close()

        search_query = goal_title_or_id
        goal_info = f"Goal: '{goal_title_or_id}'"
        if goal_row:
            goal_info = f"Goal [{goal_row[0]}]: '{goal_row[1]}' ({goal_row[2] or 'No description'})"
            search_query = f"{goal_row[1]} {goal_row[2] or ''}"

        search_results = search_research_papers(query=search_query, top_k=5)
        return f"### Learning Goal Context\n{goal_info}\n\n### Matching Recommended Papers\n{search_results}"
    except Exception as e:
        logger.exception("find_papers_for_goal failed")
        return f"Error matching papers for goal: {str(e)}"

@mcp.tool
def summarize_and_compare_papers(paper_ids: List[str]) -> str:
    """
    Retrieves full metadata, abstracts, and user notes for 2 to 5 paper IDs for comparative research analysis.
    """
    if not paper_ids:
        return "Please provide at least one paper_id to summarize or compare."
    
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        placeholders = ','.join(['%s'] * len(paper_ids))
        sql = f"""
            SELECT p.paper_id, p.title, p.abstract_text, p.publication_year, p.citation_count, p.doi, p.open_access_url
            FROM papers p
            WHERE p.paper_id IN ({placeholders})
        """
        cursor.execute(sql, paper_ids)
        rows = cursor.fetchall()
        
        # Fetch associated user notes
        cursor.execute(
            f"SELECT paper_id, content FROM notes WHERE paper_id IN ({placeholders})", paper_ids
        )
        note_rows = cursor.fetchall()
        notes_by_paper = {n[0]: n[1] for n in note_rows}
        
        cursor.close()
        conn.close()

        if not rows:
            return "None of the specified paper IDs were found in Lakebase."

        output = ["### Paper Comparison Data\n"]
        for r in rows:
            pid, title, abstract, year, citations, doi, url = r
            user_note = notes_by_paper.get(pid, "No user notes recorded.")
            output.append(
                f"--- \n"
                f"**ID:** `[{pid}]` \n"
                f"**Title:** {title} ({year}) \n"
                f"**Citations:** {citations} | **DOI:** {doi or 'N/A'} | **URL:** {url or 'N/A'} \n"
                f"**Abstract:** {abstract or 'No abstract text available.'} \n"
                f"**User Notes:** {user_note}"
            )
        return "\n\n".join(output)
    except Exception as e:
        logger.exception("summarize_and_compare_papers failed")
        return f"Error comparing papers: {str(e)}"

@mcp.tool
def generate_sequenced_reading_plan(user_id: str, goal_id: str, plan_title: str, paper_ids: List[str]) -> dict:
    """
    Generates and saves a structured, sequenced reading plan for a student's learning goal.
    """
    if not paper_ids:
        return {"status": "error", "message": "paper_ids list cannot be empty."}

    plan_id = f"plan_{uuid.uuid4().hex[:12]}"
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        sql = """
            INSERT INTO reading_plans (plan_id, user_id, goal_id, title, sequenced_paper_ids, status)
            VALUES (%s, %s, %s, %s, %s::jsonb, 'ACTIVE')
        """
        cursor.execute(sql, (plan_id, user_id, goal_id or None, plan_title, json.dumps(paper_ids)))
        conn.commit()
        
        # Optionally set reading status of first paper to READING and rest to TO_READ
        for idx, pid in enumerate(paper_ids):
            st = 'READING' if idx == 0 else 'TO_READ'
            prog_id = f"prog_{user_id}_{pid[:10]}"
            cursor.execute(
                """
                INSERT INTO reading_progress (progress_id, user_id, paper_id, status)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, paper_id) DO NOTHING
                """,
                (prog_id, user_id, pid, st)
            )
        conn.commit()
        return {
            "status": "success",
            "plan_id": plan_id,
            "title": plan_title,
            "sequenced_papers_count": len(paper_ids),
            "paper_ids": paper_ids,
            "message": f"Successfully created reading plan '{plan_title}' with {len(paper_ids)} papers."
        }
    except Exception as e:
        logger.exception("generate_sequenced_reading_plan failed")
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()
        conn.close()

@mcp.tool
def add_paper_to_collection(user_id: str, collection_name: str, paper_id: str) -> dict:
    """
    Adds a paper to a user's collection, creating the collection if it doesn't already exist.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        # Find or create collection
        cursor.execute(
            "SELECT collection_id FROM collections WHERE user_id = %s AND name ILIKE %s LIMIT 1",
            (user_id, collection_name)
        )
        coll_row = cursor.fetchone()
        if coll_row:
            coll_id = coll_row[0]
        else:
            coll_id = f"coll_{uuid.uuid4().hex[:12]}"
            cursor.execute(
                "INSERT INTO collections (collection_id, user_id, name, description) VALUES (%s, %s, %s, %s)",
                (coll_id, user_id, collection_name, f"Collection for {collection_name}")
            )
        
        cursor.execute(
            """
            INSERT INTO collection_papers (collection_id, paper_id)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
            """,
            (coll_id, paper_id)
        )
        conn.commit()
        return {
            "status": "success",
            "collection_id": coll_id,
            "collection_name": collection_name,
            "paper_id": paper_id,
            "message": f"Added paper '{paper_id}' to collection '{collection_name}'."
        }
    except Exception as e:
        logger.exception("add_paper_to_collection failed")
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()
        conn.close()

@mcp.tool
def track_progress_and_recommend(user_id: str, goal_id: str = "") -> str:
    """
    Tracks a student's current reading progress across papers and recommends the next logical paper to read.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        sql = """
            SELECT rp.paper_id, p.title, rp.status, rp.updated_at
            FROM reading_progress rp
            JOIN papers p ON rp.paper_id = p.paper_id
            WHERE rp.user_id = %s
            ORDER BY rp.updated_at DESC
        """
        cursor.execute(sql, (user_id,))
        progress_rows = cursor.fetchall()

        # Check for active reading plan
        plan_sql = "SELECT plan_id, title, sequenced_paper_ids FROM reading_plans WHERE user_id = %s AND status = 'ACTIVE' ORDER BY created_at DESC LIMIT 1"
        cursor.execute(plan_sql, (user_id,))
        plan_row = cursor.fetchone()

        cursor.close()
        conn.close()

        completed = [r for r in progress_rows if r[2] == 'COMPLETED']
        reading = [r for r in progress_rows if r[2] == 'READING']
        to_read = [r for r in progress_rows if r[2] == 'TO_READ']

        output = [f"### Reading Progress Summary for User: {user_id}\n"]
        output.append(f"• **Completed Papers ({len(completed)}):** " + (", ".join([f"`[{r[0]}]` {r[1]}" for r in completed]) if completed else "None"))
        output.append(f"• **Currently Reading ({len(reading)}):** " + (", ".join([f"`[{r[0]}]` {r[1]}" for r in reading]) if reading else "None"))
        output.append(f"• **To Read Queue ({len(to_read)}):** " + (", ".join([f"`[{r[0]}]` {r[1]}" for r in to_read]) if to_read else "None"))

        if plan_row:
            plan_id, plan_title, seq_pids = plan_row
            output.append(f"\n### Active Sequenced Reading Plan: '{plan_title}' (`{plan_id}`)")
            output.append(f"Paper sequence: {seq_pids}")

        # Recommendation logic
        recommendation = ""
        if reading:
            recommendation = f"Continue reading your active paper: `[{reading[0][0]}]` \"{reading[0][1]}\"."
        elif to_read:
            recommendation = f"Next recommended paper to read: `[{to_read[0][0]}]` \"{to_read[0][1]}\"."
        elif plan_row and seq_pids:
            recommendation = f"Next recommended paper from active plan '{plan_title}': `[{seq_pids[0]}]`."
        else:
            recommendation = "No papers currently queued. Search for papers matching your learning goal to get started!"

        output.append(f"\n💡 **Recommendation:** {recommendation}")
        return "\n".join(output)
    except Exception as e:
        logger.exception("track_progress_and_recommend failed")
        return f"Error tracking progress: {str(e)}"

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