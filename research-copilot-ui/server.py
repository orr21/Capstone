import os
import sys
import re
import json
import logging
import asyncio
import httpx
import pg8000.dbapi
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import uuid
from typing import List, Dict, Any, Optional
from fastmcp import Client as FastMCPClient
from databricks.sdk import WorkspaceClient
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
import mlflow
from mlflow.entities import SpanType

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("ui_backend")


def _init_mlflow():
    """Configure MLflow GenAI tracing for the agent flow.

    On Databricks Apps MLFLOW_EXPERIMENT_ID is auto-injected as an env var.
    Any failure only disables tracing; it never blocks the app from serving.
    """
    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "databricks"))
        experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
        if experiment_id:
            mlflow.set_experiment(experiment_id=experiment_id)
            logger.info("MLflow tracing enabled for experiment_id='%s'", experiment_id)
        else:
            logger.warning("MLFLOW_EXPERIMENT_ID not set — traces will not be saved")
            return
        mlflow.openai.autolog()
    except Exception as e:
        logger.warning("MLflow init failed, continuing without tracing: %s", e)


_init_mlflow()

app = FastAPI(title="Research Copilot Streaming Portal")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Configuration
DB_HOST = os.environ.get("DB_HOST", "ep-withered-breeze-d8845p1k.database.us-east-2.cloud.databricks.com")
DB_NAME = os.environ.get("DB_NAME", "databricks_postgres")
DB_USER = os.environ.get("DB_USER", "research-copilot-agent")
DB_PASS = os.environ.get("DB_PASS", "REDACTED")
ENDPOINT_NAME = os.environ.get("AGENT_ENDPOINT_NAME", "databricks-meta-llama-3-3-70b-instruct")

TOOL_TIMEOUT_SECONDS = 120

DEFAULT_MCP_CONFIG = {
    "lakebase": os.environ.get("MCP_SERVER_URL", "https://mcp-reserach-copilot-tools-7474657332212776.aws.databricksapps.com/mcp")
}
MCP_SERVERS: Dict[str, str] = json.loads(os.environ.get("MCP_SERVERS_CONFIG", json.dumps(DEFAULT_MCP_CONFIG)))

_w = None

def get_workspace_client():
    global _w
    if _w is None:
        _w = WorkspaceClient()
    return _w

def _get_databricks_chat_model() -> ChatOpenAI:
    """Build a LangChain ChatOpenAI pointed at the Databricks model serving endpoint."""
    w = get_workspace_client()
    headers = w.config.authenticate()
    token = headers.get("Authorization", "").replace("Bearer ", "")
    host = w.config.host.rstrip("/")
    return ChatOpenAI(
        model=ENDPOINT_NAME,
        base_url=f"{host}/serving-endpoints",
        api_key=token,
        timeout=180,
    )

def _resolve_bearer_token(forwarded_token: str = "") -> str:
    """Resolve the best available bearer token for calling other Databricks Apps.

    Priority:
    1. X-Forwarded-Access-Token injected by Databricks Apps — the user's own
       OAuth token, accepted by any app the user has access to.  No service
       principal CAN USE grant is required.
    2. WorkspaceClient M2M token — the app service principal's OAuth token.
       Requires the SP to have CAN USE on the target app.
    """
    token = (forwarded_token or "").strip()
    if token:
        logger.debug("MCP auth: using X-Forwarded-Access-Token (user OBO token)")
        return token

    # Fallback: service-principal M2M via auto-injected client_id/secret
    try:
        w = WorkspaceClient()
        headers = w.config.authenticate()
        sp_token = headers.get("Authorization", "").replace("Bearer ", "").strip()
        if sp_token:
            logger.debug("MCP auth: using WorkspaceClient M2M token (SP credentials)")
            return sp_token
    except Exception as e:
        logger.warning("WorkspaceClient M2M token fetch failed: %s", e)

    logger.warning("MCP auth: no token available — request will likely be rejected (401)")
    return ""


class _BearerAuth(httpx.Auth):
    """httpx auth that injects a pre-resolved Bearer token."""

    def __init__(self, token: str):
        self._token = token

    def auth_flow(self, request):
        if self._token:
            request.headers["Authorization"] = f"Bearer {self._token}"
        yield request


class MultiMCPRegistry:
    def __init__(self, servers: Dict[str, str]):
        self.servers = servers
        self.last_errors: List[str] = []

    def _make_auth(self, bearer_token: str) -> httpx.Auth:
        """Build a per-request auth object from the resolved token."""
        return _BearerAuth(bearer_token)

    async def get_all_tools(
        self, bearer_token: str = ""
    ) -> tuple[List[Dict[str, Any]], Dict[str, tuple[str, str]]]:
        openai_tools = []
        tool_router: Dict[str, tuple[str, str]] = {}
        self.last_errors = []
        auth = self._make_auth(bearer_token)

        for namespace, url in self.servers.items():
            try:
                async with FastMCPClient(url, auth=auth) as client:
                    tools = await client.list_tools()
                    for tool in tools:
                        namespaced_name = f"{namespace}_{tool.name}"
                        tool_router[namespaced_name] = (namespace, tool.name)
                        schema = (
                            getattr(tool, "parameters", None)
                            or getattr(tool, "inputSchema", None)
                            or {"type": "object", "properties": {}}
                        )
                        openai_tools.append({
                            "type": "function",
                            "function": {
                                "name": namespaced_name,
                                "description": f"[{namespace.upper()}] {tool.description or ''}",
                                "parameters": schema
                            }
                        })
            except Exception as e:
                msg = f"{namespace} ({url}): {e}"
                self.last_errors.append(msg)
                logger.error("Failed to load MCP server: %s", msg)

        return openai_tools, tool_router

    async def call_tool(
        self,
        namespaced_name: str,
        args: dict,
        tool_router: Dict[str, tuple[str, str]],
        bearer_token: str = "",
    ) -> str:
        if namespaced_name not in tool_router:
            raise ValueError(f"Unknown tool: {namespaced_name}")

        namespace, original_tool_name = tool_router[namespaced_name]
        
        # Pre-hook: Validate tool input guardrails before execution
        _validate_tool_call_guardrails(original_tool_name, args)

        url = self.servers[namespace]
        auth = self._make_auth(bearer_token)

        async with FastMCPClient(url, auth=auth) as client:
            result = await client.call_tool(original_tool_name, args)
            return str(result)

mcp_registry = MultiMCPRegistry(MCP_SERVERS)

def get_db_conn():
    return pg8000.dbapi.connect(
        host=DB_HOST, port=5432, database=DB_NAME,
        user=DB_USER, password=DB_PASS, ssl_context=True
    )

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    conversation_id: Optional[str] = None
    user_email: Optional[str] = None

class ConversationCreate(BaseModel):
    user_email: str
    title: Optional[str] = "New Research Chat"

@app.get("/api/conversations")
def list_conversations(user_email: str):
    """List all chat conversations for a given user email."""
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT conversation_id, title, created_at, updated_at FROM conversations WHERE user_id = %s ORDER BY updated_at DESC",
            (user_email,)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [
            {
                "conversation_id": r[0],
                "title": r[1] or "Research Session",
                "created_at": r[2].isoformat() if r[2] else "",
                "updated_at": r[3].isoformat() if r[3] else ""
            } for r in rows
        ]
    except Exception as e:
        logger.warning("Failed to list conversations: %s", e)
        return []

@app.post("/api/conversations")
def create_conversation(req: ConversationCreate):
    """Create a new conversation session for a user."""
    conv_id = f"conv_{uuid.uuid4().hex[:12]}"
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        # Ensure user exists
        cursor.execute(
            "INSERT INTO users (user_id, email, name) VALUES (%s, %s, %s) ON CONFLICT (user_id) DO NOTHING",
            (req.user_email, req.user_email, req.user_email.split('@')[0])
        )
        cursor.execute(
            "INSERT INTO conversations (conversation_id, user_id, title) VALUES (%s, %s, %s)",
            (conv_id, req.user_email, req.title)
        )
        conn.commit()
        cursor.close()
        conn.close()
        return {"conversation_id": conv_id, "title": req.title, "user_email": req.user_email}
    except Exception as e:
        logger.exception("Failed to create conversation")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations/{conversation_id}/messages")
def get_conversation_messages(conversation_id: str):
    """Get all saved messages for a conversation session."""
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT message_id, role, content, citations, created_at FROM messages WHERE conversation_id = %s ORDER BY created_at ASC",
            (conversation_id,)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [
            {
                "message_id": r[0],
                "role": r[1],
                "content": r[2],
                "citations": r[3] or [],
                "created_at": r[4].isoformat() if r[4] else ""
            } for r in rows
        ]
    except Exception as e:
        logger.warning("Failed to get messages: %s", e)
        return []

class GoalActivateRequest(BaseModel):
    user_email: str
    title: str
    description: Optional[str] = ""

@app.get("/api/goals")
def list_user_goals(user_email: str):
    """List all learning goals for a user and highlight active goal."""
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT goal_id, title, description, is_active FROM learning_goals WHERE user_id = %s ORDER BY created_at DESC",
            (user_email,)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [
            {"goal_id": r[0], "title": r[1], "description": r[2] or "", "is_active": bool(r[3])}
            for r in rows
        ]
    except Exception as e:
        logger.warning("Failed to list goals: %s", e)
        return []

@app.post("/api/goals/activate")
def activate_or_create_goal(req: GoalActivateRequest):
    """Manually activate or create a learning goal for a user, adapting recommendations and reading plans."""
    try:
        conn = get_db_conn()
        cursor = conn.cursor()

        # Deactivate previous active goals
        cursor.execute("UPDATE learning_goals SET is_active = FALSE WHERE user_id = %s", (req.user_email,))
        
        # Check if goal already exists
        cursor.execute(
            "SELECT goal_id FROM learning_goals WHERE user_id = %s AND title ILIKE %s LIMIT 1",
            (req.user_email, req.title)
        )
        existing = cursor.fetchone()
        if existing:
            goal_id = existing[0]
            cursor.execute("UPDATE learning_goals SET is_active = TRUE WHERE goal_id = %s", (goal_id,))
        else:
            goal_id = f"goal_{uuid.uuid4().hex[:12]}"
            cursor.execute(
                "INSERT INTO learning_goals (goal_id, user_id, title, description, is_active) VALUES (%s, %s, %s, %s, TRUE)",
                (goal_id, req.user_email, req.title, req.description)
            )
        
        conn.commit()
        cursor.close()
        conn.close()
        return {"status": "success", "goal_id": goal_id, "title": req.title}
    except Exception as e:
        logger.exception("Failed to activate goal")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard")
def get_user_dashboard(user_email: str):
    """Retrieve learning goals, active plan, recommendations, and reading progress for user."""
    try:
        conn = get_db_conn()
        cursor = conn.cursor()

        # 0. Fetch user learning goals
        cursor.execute(
            "SELECT goal_id, title, description, is_active FROM learning_goals WHERE user_id = %s ORDER BY is_active DESC, created_at DESC",
            (user_email,)
        )
        goal_rows = cursor.fetchall()
        goals = [{"goal_id": r[0], "title": r[1], "description": r[2] or "", "is_active": bool(r[3])} for r in goal_rows]
        active_goal = goals[0] if goals else None

        # 1. Fetch active reading plan
        cursor.execute(
            "SELECT plan_id, title, sequenced_paper_ids, created_at FROM reading_plans WHERE user_id = %s AND status = 'ACTIVE' ORDER BY created_at DESC LIMIT 1",
            (user_email,)
        )
        plan_row = cursor.fetchone()
        active_plan = None
        plan_paper_ids = []
        if plan_row:
            plan_paper_ids = plan_row[2] if isinstance(plan_row[2], list) else json.loads(plan_row[2] or "[]")
            active_plan = {
                "plan_id": plan_row[0],
                "title": plan_row[1],
                "paper_ids": plan_paper_ids,
                "created_at": plan_row[3].isoformat() if plan_row[3] else ""
            }

        # 2. Fetch reading progress counts & details
        cursor.execute(
            "SELECT rp.paper_id, p.title, rp.status FROM reading_progress rp JOIN papers p ON rp.paper_id = p.paper_id WHERE rp.user_id = %s",
            (user_email,)
        )
        prog_rows = cursor.fetchall()
        
        completed = [r for r in prog_rows if r[2] == 'COMPLETED']
        reading = [r for r in prog_rows if r[2] == 'READING']
        to_read = [r for r in prog_rows if r[2] == 'TO_READ']
        completed_papers = [{"paper_id": r[0], "title": r[1]} for r in completed]
        reading_papers = [{"paper_id": r[0], "title": r[1]} for r in reading]
        to_read_papers = [{"paper_id": r[0], "title": r[1]} for r in to_read]

        # 3. Next Recommended Paper
        recommended_paper = None
        if reading:
            recommended_paper = {"paper_id": reading[0][0], "title": reading[0][1], "status": "READING", "reason": "Currently in progress"}
        elif to_read:
            recommended_paper = {"paper_id": to_read[0][0], "title": to_read[0][1], "status": "TO_READ", "reason": "Next in queue"}
        elif plan_paper_ids:
            cursor.execute("SELECT paper_id, title, abstract_text FROM papers WHERE paper_id = %s LIMIT 1", (plan_paper_ids[0],))
            p_row = cursor.fetchone()
            if p_row:
                recommended_paper = {"paper_id": p_row[0], "title": p_row[1], "status": "PLAN_NEXT", "reason": f"First in active plan '{active_plan['title']}'"}

        cursor.close()
        conn.close()

        return {
            "user_email": user_email,
            "goals": goals,
            "active_goal": active_goal,
            "active_plan": active_plan,
            "recommended_paper": recommended_paper,
            "completed_papers": completed_papers,
            "reading_papers": reading_papers,
            "to_read_papers": to_read_papers,
            "stats": {
                "completed": len(completed),
                "reading": len(reading),
                "to_read": len(to_read)
            }
        }
    except Exception as e:
        logger.warning("Failed to fetch dashboard: %s", e)
        return {
            "user_email": user_email,
            "goals": [],
            "active_goal": None,
            "active_plan": None,
            "recommended_paper": None,
            "completed_papers": [],
            "reading_papers": [],
            "to_read_papers": [],
            "stats": {"completed": 0, "reading": 0, "to_read": 0}
        }

def _validate_tool_call_guardrails(tool_name: str, args: dict):
    """Sanitizes tool arguments without throwing runtime errors, allowing LLM to handle empty queries naturally."""
    if not isinstance(args, dict):
        return
    
    if tool_name == "search_research_papers":
        query = str(args.get("query", "")).strip()
        if not query:
            args["query"] = "research"
        top_k = args.get("top_k", 5)
        if isinstance(top_k, int):
            args["top_k"] = max(1, min(top_k, 20))
            
    elif tool_name == "summarize_and_compare_papers":
        # Drop placeholder paper IDs (e.g. "<paper_id_1>") the model may hallucinate
        raw_ids = args.get("paper_ids")
        if isinstance(raw_ids, list):
            cleaned = [pid for pid in raw_ids if not _is_placeholder_paper_id(str(pid))]
            args["paper_ids"] = cleaned if cleaned else ["__placeholders_only__"]
            
    elif tool_name == "update_reading_progress":
        status = str(args.get("status", "")).upper()
        if status in ["TO_READ", "READING", "COMPLETED"]:
            args["status"] = status


def _is_placeholder_paper_id(paper_id: str) -> bool:
    """True if a paper ID looks like an LLM placeholder rather than a real OpenAlex ID."""
    pid = (paper_id or "").strip()
    if not pid:
        return True
    if re.match(r"^<[^>]+>$|^paper_id_\d+$|^PAPER_ID_\d+$", pid):
        return True
    if "<" in pid or ">" in pid:
        return True
    if pid.lower().startswith("paper_id") and not pid.lower().startswith("https"):
        return True
    return False


_ERROR_PATTERNS = (
    '"status": "error"',
    "error comparing",
    "error matching",
    "unknown tool",
    "not found",
    "none of the specified",
    "placeholder",
    "no research papers found",
    "error:",
)


def _recovery_nudge(tool_results: list) -> str:
    """Build a corrective user message when a tool call failed or hit a dead-end.

    `tool_results` is a list of (tool_name, raw_output_string). Returns an
    empty string when no corrective action is needed.
    """
    failed_tool = None
    for tool_name, raw_output in tool_results:
        output = str(raw_output or "").lower()
        if any(p in output for p in _ERROR_PATTERNS):
            failed_tool = tool_name
            break
    if not failed_tool:
        return ""

    if "compare" in failed_tool.lower():
        return (
            "A tool call failed: `summarize_and_compare_papers` was given paper IDs that were "
            "placeholders or not in Lakebase. Run `search_research_papers` or `find_papers_for_goal` "
            "FIRST to obtain real OpenAlex paper IDs (e.g. https://openalex.org/W...), then retry "
            "`summarize_and_compare_papers` with those real IDs. If you cannot get real IDs, answer "
            "the user directly with what you know and stop calling tools."
        )

    return (
        f"The tool call `{failed_tool}` failed. Review the tool error above, fix the arguments, "
        "and retry the call. If the tool cannot fulfill the request, answer the user directly "
        "with what you know and stop calling tools."
    )

@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Delete a conversation session."""
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM conversations WHERE conversation_id = %s", (conversation_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return {"status": "success", "message": f"Deleted conversation {conversation_id}"}
    except Exception as e:
        logger.exception("Failed to delete conversation")
        raise HTTPException(status_code=500, detail=str(e))

def save_chat_message(conversation_id: str, role: str, content: str):
    """Helper to record a chat message into Lakebase PostgreSQL."""
    if not conversation_id or not content:
        return
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (message_id, conversation_id, role, content) VALUES (%s, %s, %s, %s)",
            (msg_id, conversation_id, role, content)
        )
        cursor.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE conversation_id = %s",
            (conversation_id,)
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.warning("Could not save message to database: %s", e)

@app.get("/")
def read_root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="index.html not found")

TOOL_CATEGORIES = {
    "search_research_papers": ("Search & Retrieval", "🔍", "Perform semantic vector search across research papers in Lakebase."),
    "find_papers_for_goal": ("Search & Retrieval", "🎯", "Match papers to student learning goals and retrieve relevant context."),
    "create_or_set_learning_goal": ("Planning & Progress", "🚀", "Create a new learning goal or switch your active study topic."),
    "summarize_and_compare_papers": ("Analysis & Comparison", "⚖️", "Fetch raw abstracts to compare 2+ papers."),
    "count_papers": ("Analysis & Comparison", "🔢", "Count research papers matching keywords or publication year."),
    "generate_sequenced_reading_plan": ("Planning & Progress", "🗺️", "Generate and save a sequenced reading plan for a learning goal."),
    "track_progress_and_recommend": ("Planning & Progress", "💡", "Track reading progress and recommend the next logical paper."),
    "update_reading_progress": ("Planning & Progress", "📌", "Update paper reading status (TO_READ, READING, COMPLETED).")
}

@app.get("/api/config")
def read_config():
    return {"model": ENDPOINT_NAME, "mcp_servers": list(MCP_SERVERS.keys())}

@app.get("/api/tools")
async def read_tools(request: Request):
    forwarded_token = request.headers.get("x-forwarded-access-token", "")
    bearer_token = _resolve_bearer_token(forwarded_token)
    
    openai_tools, tool_router = await mcp_registry.get_all_tools(bearer_token=bearer_token)
    
    grouped = {}
    for t in openai_tools:
        ns_name = t["function"]["name"]
        ns, orig_name = tool_router.get(ns_name, ("lakebase", ns_name))
        cat, icon, desc = TOOL_CATEGORIES.get(
            orig_name, ("General Tools", "🛠️", t["function"].get("description", ""))
        )
        
        if ns not in grouped:
            grouped[ns] = {}
        if cat not in grouped[ns]:
            grouped[ns][cat] = []
            
        grouped[ns][cat].append({
            "namespaced_name": ns_name,
            "original_name": orig_name,
            "description": desc,
            "icon": icon,
            "parameters": t["function"].get("parameters", {})
        })
        
    return {"mcp_servers": grouped, "total_tools": len(openai_tools)}

@app.get("/api/kanban/plans")
def list_kanban_plans(user_email: str):
    """List the user's reading plans to populate Kanban boards."""
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT plan_id, title, status, created_at FROM reading_plans WHERE user_id = %s ORDER BY created_at DESC",
            (user_email,)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return {
            "plans": [
                {"plan_id": r[0], "title": r[1], "status": r[2], "created_at": r[3].isoformat() if r[3] else ""}
                for r in rows
            ]
        }
    except Exception as e:
        logger.warning("Failed to list kanban plans: %s", e)
        return {"plans": []}

@app.get("/api/kanban/board")
def get_kanban_board(user_email: str, plan_id: str = ""):
    """Return a reading plan's papers bucketed by reading_progress status."""
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        if plan_id:
            cursor.execute(
                "SELECT plan_id, title, sequenced_paper_ids FROM reading_plans WHERE user_id = %s AND plan_id = %s LIMIT 1",
                (user_email, plan_id)
            )
        else:
            cursor.execute(
                "SELECT plan_id, title, sequenced_paper_ids FROM reading_plans WHERE user_id = %s AND status = 'ACTIVE' ORDER BY created_at DESC LIMIT 1",
                (user_email,)
            )
        plan_row = cursor.fetchone()
        if not plan_row:
            cursor.close()
            conn.close()
            return {"plan": None, "to_read": [], "reading": [], "completed": []}

        plan_id_val, plan_title, seq = plan_row
        paper_ids = seq if isinstance(seq, list) else json.loads(seq or "[]")

        status_map = {}  # paper_id -> (title, status)
        if paper_ids:
            placeholders = ','.join(['%s'] * len(paper_ids))
            cursor.execute(
                f"SELECT rp.paper_id, p.title, rp.status FROM reading_progress rp JOIN papers p ON rp.paper_id = p.paper_id WHERE rp.user_id = %s AND rp.paper_id IN ({placeholders})",
                (user_email, *paper_ids)
            )
            rows = cursor.fetchall()
            for r in rows:
                status_map[r[0]] = (r[1], r[2])

        cursor.close()
        conn.close()

        to_read, reading, completed = [], [], []
        for pid in paper_ids:
            title_v, status_v = status_map.get(pid, ("", "TO_READ"))
            item = {"paper_id": pid, "title": title_v}
            if status_v == 'COMPLETED':
                completed.append(item)
            elif status_v == 'READING':
                reading.append(item)
            else:
                to_read.append(item)

        return {
            "plan": {"plan_id": plan_id_val, "title": plan_title},
            "to_read": to_read,
            "reading": reading,
            "completed": completed,
        }
    except Exception as e:
        logger.exception("get_kanban_board failed")
        return {"plan": None, "to_read": [], "reading": [], "completed": []}

class KanbanChatSaveRequest(BaseModel):
    user_email: str
    messages: List[Dict[str, str]]

@app.get("/api/kanban/chat/messages")
def list_kanban_chat_messages(user_email: str):
    """Return the user's persisted Reading Assistant chat (no session grouping)."""
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM kanban_chat_messages WHERE user_id = %s ORDER BY created_at ASC LIMIT 200",
            (user_email,)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return {"messages": [{"role": r[0], "content": r[1]} for r in rows]}
    except Exception as e:
        logger.warning("Failed to list kanban chat messages: %s", e)
        return {"messages": []}

@app.post("/api/kanban/chat/messages")
def save_kanban_chat_messages(req: KanbanChatSaveRequest):
    """Persist Reading Assistant chat messages for a user (quick conversations, no sessions)."""
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        for m in req.messages:
            role = str(m.get("role", "user"))
            content = str(m.get("content", "")).strip()
            if not content:
                continue
            mid = f"kmsg_{uuid.uuid4().hex[:12]}"
            cursor.execute(
                "INSERT INTO kanban_chat_messages (message_id, user_id, role, content) VALUES (%s, %s, %s, %s)",
                (mid, req.user_email, role, content)
            )
        conn.commit()
        cursor.close()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        logger.exception("Failed to save kanban chat messages")
        return {"status": "error", "message": str(e)}

class ReadingProgressRequest(BaseModel):
    user_email: str
    paper_id: str
    status: str

@app.post("/api/reading_progress")
def set_reading_progress(req: ReadingProgressRequest):
    """Directly update a paper's reading status (TO_READ / READING / COMPLETED).

    Accepts a full OpenAlex URL or a bare ID; resolves to the stored paper_id first.
    """
    status_clean = req.status.upper()
    if status_clean not in ['TO_READ', 'READING', 'COMPLETED']:
        raise HTTPException(status_code=400, detail="Invalid status.")
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT paper_id FROM papers WHERE paper_id = %s OR paper_id LIKE %s LIMIT 1",
            (req.paper_id, f"%/{req.paper_id}")
        )
        row = cursor.fetchone()
        if not row:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail=f"Paper '{req.paper_id}' not found in Lakebase.")
        full_id = row[0]

        progress_id = f"prog_{uuid.uuid4().hex[:12]}"
        cursor.execute(
            """
            INSERT INTO reading_progress (progress_id, user_id, paper_id, status)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id, paper_id) DO UPDATE SET status = EXCLUDED.status, updated_at = CURRENT_TIMESTAMP
            """,
            (progress_id, req.user_email, full_id, status_clean)
        )
        conn.commit()
        cursor.close()
        conn.close()
        return {"status": "success", "paper_id": full_id, "new_status": status_clean}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to update reading progress")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: ChatRequest, request: Request):
    try:
        user_email = req.user_email or request.headers.get("x-forwarded-user", "demo_user@workspace.com")
        conversation_id = req.conversation_id

        forwarded_token = request.headers.get("x-forwarded-access-token", "")
        bearer_token = _resolve_bearer_token(forwarded_token)
        if not bearer_token:
            logger.warning("No bearer token resolved; MCP calls will likely fail with 401")

        llm = _get_databricks_chat_model()
        openai_tools, tool_router = await mcp_registry.get_all_tools(bearer_token=bearer_token)

        last_user_text = req.messages[-1].content if req.messages else ""
        if conversation_id and last_user_text:
            save_chat_message(conversation_id, "user", last_user_text)

        # Query active learning goal for user
        active_goal_str = "Active Goal: General Research"
        try:
            conn = get_db_conn()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT title, description FROM learning_goals WHERE user_id = %s AND is_active = TRUE LIMIT 1",
                (user_email,)
            )
            g_row = cursor.fetchone()
            if g_row:
                active_goal_str = f"Active Goal: '{g_row[0]}'" + (f" ({g_row[1]})" if g_row[1] else "")
            cursor.close()
            conn.close()
        except Exception as ge:
            logger.warning("Failed to fetch active goal for system prompt: %s", ge)

        messages_payload = [
            {
                "role": "system",
                "content": (
                    f"You are Research Copilot, an AI research assistant helping {user_email}. "
                    f"The user's {active_goal_str}. "
                    "Always tailor your paper searches, reading plans, and recommendations to help the user achieve this active goal. "
                    "When referencing research papers in your final answer, ALWAYS include explicit citations "
                    "in the format: `[paper_id]` \"Title\" (Year) — for example `[W7172556967]` \"Learning the Pareto Frontier\" (2026). "
                    "Help the user move papers across their Kanban board: use `update_reading_progress` to mark "
                    "papers as READING when they start one and as COMPLETED when they finish it. "
                    "If the user asks to create a new study plan or switch learning goals while an active reading plan exists, "
                    "ask for user confirmation before overwriting their active study plan. "
                    "Do not invent tool names or arguments. Wait for tool results, then provide a helpful synthesis."
                )
            }
        ]
        for m in req.messages:
            messages_payload.append({"role": m.role, "content": m.content})

        agent = run_agent(user_email, messages_payload, llm, openai_tools, tool_router, bearer_token)
        return StreamingResponse(
            _sse_encoder_with_run(agent, user_email, last_user_text, conversation_id),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.exception("Failed to initialize chat stream")
        async def _error_stream():
            err_data = {
                "type": "error",
                "content": "Failed to initialize agent connection.",
                "detail": str(e),
                "hint": "Check backend configuration and logs."
            }
            yield f"data: {json.dumps(err_data)}\n\n"
        return StreamingResponse(_error_stream(), media_type="text/event-stream")


def _aggregate_agent_events(events):
    """Collapse streamed SSE events into a single span output for MLflow.

    Captures: final answer text, each tool call with its name/args/output,
    and any error messages. Returned dict is stored as the AGENT span output
    in the MLflow trace UI.
    """
    content_parts = []
    tool_calls = []
    tool_starts = {}  # name -> args, so we can join with tool_end
    errors = []
    for ev in events:
        etype = ev.get("type")
        if etype == "chunk":
            content_parts.append(ev.get("content", ""))
        elif etype == "tool_start":
            tool_starts[ev.get("name", "")] = ev.get("args", {})
        elif etype == "tool_end":
            name = ev.get("name", "")
            tool_calls.append({
                "name": name,
                "inputs": tool_starts.get(name, {}),
                "output": ev.get("output"),
            })
        elif etype == "error":
            errors.append(ev.get("content"))
    return {
        "final_answer": "".join(content_parts),
        "tool_calls": tool_calls,
        "errors": errors,
        "status": "error" if errors else "ok",
    }


async def _sse_encoder_with_run(events, user_email: str = "", last_user_msg: str = "", conversation_id: Optional[str] = None):
    """Wrap the agent generator in a single MLflow trace so each request is one debuggable session.

    mlflow.start_span() as the root creates ONE trace per request ('research_copilot_agent');
    every LLM call (autolog) and MCP tool call nests as a child span inside that trace, so you
    can grab a single trace link and share it. Tracing is best-effort: if it fails, events still
    stream normally.
    """
    collected: list = []
    final_text_parts: list = []

    span_cm = None
    agent_span = None
    if os.environ.get("MLFLOW_EXPERIMENT_ID"):
        try:
            span_cm = mlflow.start_span(name="research_copilot_agent", span_type=SpanType.AGENT)
            agent_span = span_cm.__enter__()
            try:
                agent_span.set_inputs({"user": user_email, "question": last_user_msg[:500]})
            except Exception:
                pass
            try:
                # Tag the trace; session_id links all turns of a conversation as one session
                mlflow.update_current_trace(
                    session_id=conversation_id,
                    user=user_email,
                    tags={"app": "research-copilot-ui", "model": ENDPOINT_NAME},
                )
            except Exception:
                pass
        except Exception as e:
            logger.warning("Failed to start MLflow trace, continuing without tracing: %s", e)
            span_cm = None
            agent_span = None

    try:
        async for event in events:
            collected.append(event)
            if event.get("type") == "chunk":
                final_text_parts.append(event.get("content", ""))
            yield f"data: {json.dumps(event)}\n\n"
    except Exception:
        # If streaming/tracing setup failed before any events, stream without tracing
        if not collected:
            async for event in events:
                if event.get("type") == "chunk":
                    final_text_parts.append(event.get("content", ""))
                yield f"data: {json.dumps(event)}\n\n"
    finally:
        if agent_span is not None:
            try:
                agent_span.set_outputs(_aggregate_agent_events(collected))
            except Exception:
                pass
            try:
                span_cm.__exit__(None, None, None)
            except Exception:
                pass

    # Save assistant message to Lakebase PostgreSQL
    full_response = "".join(final_text_parts)
    if conversation_id and full_response:
        save_chat_message(conversation_id, "assistant", full_response)



async def run_agent(user_email: str, messages_payload: list, llm, openai_tools, tool_router, bearer_token: str = ""):
    """LangChain-powered agent loop with native tool calling and SSE streaming.

    `llm` is a LangChain ChatOpenAI pointed at the Databricks serving endpoint. Tools are
    bound once via `bind_tools`, and each `AIMessage.tool_calls` round is executed through the
    existing MCP registry. Yields the same SSE event dicts the frontend already understands.
    """
    def _to_lc_messages(msgs):
        out = []
        for m in msgs:
            r = m.get("role")
            if r == "system":
                out.append(SystemMessage(content=m.get("content") or ""))
            elif r == "user":
                out.append(HumanMessage(content=m.get("content") or ""))
            elif r == "assistant":
                content = m.get("content") or ""
                tcs = m.get("tool_calls")
                if tcs:
                    lc_tcs = []
                    for tc in tcs:
                        fn = tc.get("function") or {}
                        name = fn.get("name") or tc.get("name")
                        ar = fn.get("arguments") or tc.get("args")
                        args = json.loads(ar) if isinstance(ar, str) else (ar or {})
                        lc_tcs.append({"id": tc.get("id"), "name": name, "args": args, "type": "tool_call"})
                    out.append(AIMessage(content=content, tool_calls=lc_tcs))
                else:
                    out.append(AIMessage(content=content))
            elif r == "tool":
                out.append(ToolMessage(
                    content=str(m.get("content") or ""),
                    tool_call_id=m.get("tool_call_id"),
                    name=m.get("name") or "",
                ))
        return out

    def _str_content(msg):
        c = msg.content
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "".join(p if isinstance(p, str) else (p.get("text", "") if isinstance(p, dict) else "") for p in c)
        return str(c or "")

    def _parse_text_calls(content, turn_idx):
        if not content or "<function=" not in content:
            return [], content
        matches = re.findall(r'<function=([a-zA-Z0-9_.-]+)>\s*(\{.*?\})', content, re.DOTALL)
        calls = []
        for idx, (n, a) in enumerate(matches):
            try:
                args = json.loads(a)
            except Exception:
                args = {}
            calls.append({"id": f"text_call_{turn_idx}_{idx}", "name": n, "args": args, "type": "tool_call"})
        cleaned = re.sub(r'<function=[a-zA-Z0-9_.-]+>\s*\{.*?\}', "", content, flags=re.DOTALL).strip()
        return calls, cleaned

    def _stream_chunks(chat_model, lc_messages):
        for chunk in chat_model.stream(lc_messages):
            piece = getattr(chunk, "content", "")
            if isinstance(piece, list):
                piece = "".join(p if isinstance(p, str) else (p.get("text", "") if isinstance(p, dict) else "") for p in piece)
            if piece:
                yield {"type": "chunk", "content": piece}

    try:
        yield {"type": "status", "content": "Discovering MCP tools..."}

        span = mlflow.get_current_active_span()
        if span is not None:
            try:
                span.set_attributes({
                    "mcp_tools_loaded": [t["function"]["name"] for t in openai_tools],
                    "mcp_tools_count": len(openai_tools),
                })
            except Exception:
                pass

        if not openai_tools:
            detail = "; ".join(mcp_registry.last_errors) or "no MCP servers configured"
            logger.error("No MCP tools available, skipping model call: %s", detail)
            yield {
                "type": "error",
                "content": "Could not connect to the research database tools.",
                "detail": detail,
                "hint": (
                    "The MCP server app is unreachable or is blocking this app. In Databricks, "
                    "open the MCP server app → Settings → Authentication → set it to 'Open access', "
                    "then redeploy this app."
                ),
            }
            yield {"type": "done"}
            return

        llm_with_tools = llm.bind_tools(openai_tools)
        max_turns = 5
        turn = 0

        while turn < max_turns:
            turn += 1
            yield {"type": "status", "content": f"Reasoning (step {turn})..."}

            lc_messages = _to_lc_messages(messages_payload)
            try:
                response = llm_with_tools.invoke(lc_messages)
            except Exception as llm_err:
                logger.exception("LLM invoke failed")
                yield {
                    "type": "error",
                    "content": "The model call failed.",
                    "detail": str(llm_err),
                    "hint": "Retry, and check the app logs for the full traceback.",
                }
                yield {"type": "done"}
                return

            content = _str_content(response)
            raw_tool_calls = response.tool_calls or []
            tool_calls = [{"id": tc["id"], "name": tc["name"], "args": tc.get("args", {})} for tc in raw_tool_calls] if raw_tool_calls else []

            if not tool_calls:
                tool_calls, content = _parse_text_calls(content, turn)

            if not tool_calls:
                yield {"type": "status", "content": "Synthesizing final answer..."}
                if content:
                    yield {"type": "chunk", "content": content}
                else:
                    for ev in _stream_chunks(llm, lc_messages):
                        yield ev
                yield {"type": "done"}
                return

            # Persist assistant turn in OpenAI-dict history (for persistence in DB / future turns)
            messages_payload.append({
                "role": "assistant",
                "content": content or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])},
                    }
                    for tc in tool_calls
                ],
            })

            by_original = {orig: ns_name for ns_name, (ns, orig) in tool_router.items()}
            turn_tool_results = []

            for tc in tool_calls:
                tc_id = tc["id"]
                tc_name = tc["name"]
                tc_args = tc.get("args") or {}

                resolved = tc_name
                if resolved not in tool_router and tc_name in by_original:
                    resolved = by_original[tc_name]
                if resolved not in tool_router:
                    logger.error("Model called unknown tool '%s'", tc_name)
                    original_tool_name = tc_name
                    tool_output = json.dumps({"status": "error", "message": f"Unknown tool: {tc_name}"})
                else:
                    _, original_tool_name = tool_router[resolved]
                    _validate_tool_call_guardrails(original_tool_name, tc_args)

                    yield {"type": "tool_start", "name": resolved, "args": tc_args}

                    is_retriever = any(k in original_tool_name.lower() for k in ["search", "find", "recommend", "compare"])
                    span_type_val = SpanType.RETRIEVER if is_retriever else SpanType.TOOL

                    with mlflow.start_span(
                        name=f"tool_{original_tool_name}",
                        span_type=span_type_val,
                    ) as tool_span:
                        tool_span.set_inputs({
                            "tool_name": resolved,
                            "original_tool_name": original_tool_name,
                            "arguments": tc_args,
                        })
                        try:
                            tool_output = await asyncio.wait_for(
                                mcp_registry.call_tool(
                                    resolved, tc_args, tool_router, bearer_token=bearer_token
                                ),
                                timeout=TOOL_TIMEOUT_SECONDS,
                            )
                            tool_span.set_outputs({"output": tool_output, "status": "success"})
                        except (asyncio.TimeoutError, TimeoutError) as tool_err:
                            tool_output = json.dumps({
                                "status": "error",
                                "message": f"Tool '{resolved}' timed out after {TOOL_TIMEOUT_SECONDS}s.",
                            })
                            tool_span.set_outputs({"error": str(tool_err), "status": "error"})
                            logger.warning("Tool '%s' timed out after %ss", resolved, TOOL_TIMEOUT_SECONDS)
                        except Exception as tool_err:
                            tool_output = json.dumps({"status": "error", "message": str(tool_err)})
                            tool_span.set_outputs({"error": str(tool_err), "status": "error"})
                            logger.exception("Tool '%s' raised an exception", resolved)

                    yield {"type": "tool_end", "name": resolved, "output": tool_output}

                turn_tool_results.append((original_tool_name, tool_output))
                messages_payload.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": tc_name,
                    "content": str(tool_output),
                })

            recovery = _recovery_nudge(turn_tool_results)
            if recovery:
                yield {"type": "status", "content": "Recovering from a failed tool call..."}
                messages_payload.append({"role": "user", "content": recovery})

        # Exhausted turns: force a streamed final synthesis so the user always gets an answer.
        yield {"type": "status", "content": "Synthesizing final answer..."}
        try:
            lc_messages = _to_lc_messages(messages_payload)
            for ev in _stream_chunks(llm, lc_messages):
                yield ev
        except Exception as final_err:
            logger.exception("Final synthesis call failed")
            yield {
                "type": "error",
                "content": "Something went wrong while answering.",
                "detail": str(final_err),
                "hint": "Retry, and check the app logs for the full traceback.",
            }
        yield {"type": "done"}

    except Exception as e:
        logger.exception("Stream execution error")
        yield {
            "type": "error",
            "content": "Something went wrong while answering.",
            "detail": str(e),
            "hint": "Retry, and check the app logs for the full traceback.",
        }