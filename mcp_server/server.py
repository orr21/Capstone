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

TABLES_SQL = [
    """
    CREATE TABLE IF NOT EXISTS users (
        user_id VARCHAR(255) PRIMARY KEY,
        name VARCHAR(255),
        email VARCHAR(255) UNIQUE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS conversations (
        conversation_id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        title VARCHAR(255),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        message_id VARCHAR(255) PRIMARY KEY,
        conversation_id VARCHAR(255) NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
        role VARCHAR(50) NOT NULL,
        content TEXT NOT NULL,
        citations JSONB DEFAULT '[]'::jsonb,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS learning_goals (
        goal_id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        title VARCHAR(255) NOT NULL,
        description TEXT,
        target_date DATE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS papers (
        paper_id VARCHAR(255) PRIMARY KEY,
        doi VARCHAR(255),
        title TEXT NOT NULL,
        abstract_text TEXT,
        publication_year INT,
        citation_count INT DEFAULT 0,
        open_access_url TEXT,
        topics JSONB DEFAULT '[]'::jsonb,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS authors (
        author_id VARCHAR(255) PRIMARY KEY,
        display_name VARCHAR(255),
        orcid VARCHAR(255),
        institution_name VARCHAR(255)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS paper_authors (
        paper_id VARCHAR(255) REFERENCES papers(paper_id) ON DELETE CASCADE,
        author_id VARCHAR(255) REFERENCES authors(author_id) ON DELETE CASCADE,
        author_position INT,
        PRIMARY KEY (paper_id, author_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS collections (
        collection_id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS collection_papers (
        collection_id VARCHAR(255) REFERENCES collections(collection_id) ON DELETE CASCADE,
        paper_id VARCHAR(255) REFERENCES papers(paper_id) ON DELETE CASCADE,
        added_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (collection_id, paper_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS reading_progress (
        progress_id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        paper_id VARCHAR(255) REFERENCES papers(paper_id) ON DELETE CASCADE,
        status VARCHAR(50) NOT NULL CHECK (status IN ('TO_READ', 'READING', 'COMPLETED')),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT unique_user_paper UNIQUE (user_id, paper_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS reading_plans (
        plan_id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        goal_id VARCHAR(255) REFERENCES learning_goals(goal_id) ON DELETE SET NULL,
        title VARCHAR(255) NOT NULL,
        sequenced_paper_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
        status VARCHAR(50) DEFAULT 'ACTIVE',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS notes (
        note_id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        paper_id VARCHAR(255) REFERENCES papers(paper_id) ON DELETE CASCADE,
        content TEXT NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """
]

def init_db_tables():
    """Ensure all required tables (conversations, messages, reading_plans, etc.) exist in Lakebase."""
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        for query in TABLES_SQL:
            cursor.execute(query)
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("✅ Database tables (conversations, messages, reading_plans, etc.) initialized successfully.")
    except Exception as e:
        logger.warning("Auto table initialization skipped or failed: %s", e)

# Automatically attempt schema initialization on start
init_db_tables()

@mcp.tool
def search_research_papers(query: str, top_k: int = 5) -> str:
    """
    Semantically searches research papers in Lakebase using vector similarity and hybrid keyword matching.
    Returns paper IDs, titles, publication years, citations, and abstracts.
    """
    try:
        model = get_embed_model()
        q_vec = '{' + ','.join(str(float(x)) for x in model.encode([query])[0]) + '}'
        
        conn = get_db_conn()
        cursor = conn.cursor()
        
        # 1. Hybrid Check: Direct keyword match in papers table
        kw_sql = """
            SELECT paper_id, title, abstract_text, publication_year, citation_count
            FROM papers
            WHERE title ILIKE %s OR abstract_text ILIKE %s
            LIMIT %s;
        """
        cursor.execute(kw_sql, (f"%{query}%", f"%{query}%", top_k))
        kw_rows = cursor.fetchall()

        # 2. Vector similarity search
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
        vec_rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Combine keyword matches and relevant vector matches (similarity >= 0.35)
        relevant_rows = []
        seen_pids = set()

        # Add direct keyword matches first
        for r in kw_rows:
            pid = r[0]
            seen_pids.add(pid)
            snippet = r[2][:300] + "..." if r[2] else "No abstract available."
            relevant_rows.append(
                f"• [{pid}] \"{r[1]}\" ({r[3]}) | Citations: {r[4]} | Keyword Match\n  Abstract: {snippet}"
            )

        # Add vector matches if score >= 0.35
        for r in vec_rows:
            pid = r[0]
            sim = r[5]
            if pid not in seen_pids and sim >= 0.35:
                seen_pids.add(pid)
                snippet = r[2][:300] + "..." if r[2] else "No abstract available."
                relevant_rows.append(
                    f"• [{pid}] \"{r[1]}\" ({r[3]}) | Citations: {r[4]} | Match Score: {sim:.2f}\n  Abstract: {snippet}"
                )

        if not relevant_rows:
            top_score_msg = f" (Top similarity score was {vec_rows[0][5]:.2f})" if vec_rows else ""
            return f"No relevant research papers matching '{query}' were found in Lakebase{top_score_msg}. Please try a different query or topic."

        return "\n\n".join(relevant_rows[:top_k])
    except Exception as e:
        logger.exception("Search failed")
        return f"Error executing search: {str(e)}"

@mcp.tool
def create_or_set_learning_goal(user_id: str, title: str, description: str = "") -> dict:
    """
    Creates a new learning goal for a user or activates an existing one. Allows users to switch study topics smoothly.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT goal_id, title FROM learning_goals WHERE user_id = %s AND title ILIKE %s LIMIT 1",
            (user_id, title)
        )
        existing = cursor.fetchone()
        if existing:
            goal_id = existing[0]
            msg = f"Activated existing learning goal '{existing[1]}' (`{goal_id}`)."
        else:
            goal_id = f"goal_{uuid.uuid4().hex[:12]}"
            cursor.execute(
                "INSERT INTO learning_goals (goal_id, user_id, title, description) VALUES (%s, %s, %s, %s)",
                (goal_id, user_id, title, description)
            )
            conn.commit()
            msg = f"Created new learning goal '{title}' (`{goal_id}`)."
        
        cursor.close()
        conn.close()
        return {"status": "success", "goal_id": goal_id, "title": title, "message": msg}
    except Exception as e:
        logger.exception("create_or_set_learning_goal failed")
        return {"status": "error", "message": str(e)}

@mcp.tool
def find_papers_for_goal(user_id: str, goal_title_or_id: str) -> str:
    """
    Finds research papers matching a student's learning goal. Automatically creates the goal if it does not exist.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        # Search for existing goal
        cursor.execute(
            "SELECT goal_id, title, description FROM learning_goals WHERE user_id = %s AND (goal_id = %s OR title ILIKE %s) LIMIT 1",
            (user_id, goal_title_or_id, f"%{goal_title_or_id}%")
        )
        goal_row = cursor.fetchone()
        
        if not goal_row:
            # Auto-create goal so user is never blocked
            goal_id = f"goal_{uuid.uuid4().hex[:12]}"
            cursor.execute(
                "INSERT INTO learning_goals (goal_id, user_id, title, description) VALUES (%s, %s, %s, %s)",
                (goal_id, user_id, goal_title_or_id, f"Auto-created goal for {goal_title_or_id}")
            )
            conn.commit()
            goal_info = f"New Goal Created [{goal_id}]: '{goal_title_or_id}'"
            search_query = goal_title_or_id
        else:
            goal_info = f"Goal [{goal_row[0]}]: '{goal_row[1]}' ({goal_row[2] or 'No description'})"
            search_query = f"{goal_row[1]} {goal_row[2] or ''}"

        cursor.close()
        conn.close()

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
def generate_sequenced_reading_plan(user_id: str, goal_id: str = "", plan_title: str = "Sequenced Reading Plan", paper_ids: List[str] = []) -> dict:
    """
    Generates and saves a structured, sequenced reading plan for a student's learning goal.
    If paper_ids is empty, automatically retrieves and sequences top relevant papers from Lakebase.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        
        # If paper_ids is empty, automatically query matching papers from Lakebase
        if not paper_ids:
            search_query = "research"
            if goal_id:
                cursor.execute(
                    "SELECT title, description FROM learning_goals WHERE goal_id = %s OR user_id = %s LIMIT 1",
                    (goal_id, user_id)
                )
                g_row = cursor.fetchone()
                if g_row:
                    search_query = f"{g_row[0]} {g_row[1] or ''}"
            
            model = get_embed_model()
            q_vec = '{' + ','.join(str(float(x)) for x in model.encode([search_query])[0]) + '}'
            cursor.execute(
                """
                SELECT p.paper_id FROM paper_embeddings pe
                JOIN papers p ON pe.paper_id = p.paper_id
                ORDER BY pe.embedding <=> %s::double precision[]::vector ASC
                LIMIT 5
                """,
                (q_vec,)
            )
            p_rows = cursor.fetchall()
            paper_ids = [r[0] for r in p_rows]
            
            # Fallback if paper_embeddings table is empty
            if not paper_ids:
                cursor.execute("SELECT paper_id FROM papers LIMIT 5")
                paper_ids = [r[0] for r in cursor.fetchall()]

        if not paper_ids:
            return {"status": "error", "message": "No research papers found in Lakebase to create a reading plan."}

        plan_id = f"plan_{uuid.uuid4().hex[:12]}"
        sql = """
            INSERT INTO reading_plans (plan_id, user_id, goal_id, title, sequenced_paper_ids, status)
            VALUES (%s, %s, %s, %s, %s::jsonb, 'ACTIVE')
        """
        cursor.execute(sql, (plan_id, user_id, goal_id or None, plan_title, json.dumps(paper_ids)))
        conn.commit()
        
        # Set reading status of first paper to READING and rest to TO_READ
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