import os
import sys
import json
import logging
import asyncio
import httpx
import pg8000.dbapi
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from openai import OpenAI
from fastmcp import Client as FastMCPClient
from databricks.sdk import WorkspaceClient
import mlflow
from mlflow.entities import SpanType

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("ui_backend")


def _init_mlflow():
    """Configure MLflow GenAI tracing for the agent flow.

    On Databricks Apps this logs traces to the configured experiment.
    Any failure only disables tracing; it never blocks the app from serving.
    """
    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "databricks"))
        experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "/Shared/research-copilot-tracing")
        
        success = False
        if experiment_id:
            try:
                mlflow.set_experiment(experiment_id=experiment_id)
                logger.info("MLflow tracing enabled for experiment_id='%s'", experiment_id)
                success = True
            except Exception as exp_err:
                logger.warning("Could not set experiment by ID '%s' (%s), trying name fallback...", experiment_id, exp_err)
        
        if not success:
            try:
                mlflow.set_experiment(experiment_name)
                logger.info("MLflow tracing enabled for experiment_name='%s'", experiment_name)
                success = True
            except Exception as name_err:
                logger.warning("Could not set experiment by name '%s': %s", experiment_name, name_err)

        if success:
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

DEFAULT_MCP_CONFIG = {
    "lakebase": os.environ.get("MCP_SERVER_URL", "https://mcp-reserach-copilot-tools-7474657332212776.aws.databricksapps.com/mcp")
}
MCP_SERVERS: Dict[str, str] = json.loads(os.environ.get("MCP_SERVERS_CONFIG", json.dumps(DEFAULT_MCP_CONFIG)))

_w = None
_openai_client = None

def get_workspace_client():
    global _w
    if _w is None:
        _w = WorkspaceClient()
    return _w

def get_openai_client():
    w = get_workspace_client()
    headers = w.config.authenticate()
    token = headers.get("Authorization", "").replace("Bearer ", "")
    host = w.config.host.rstrip("/")
    return OpenAI(api_key=token, base_url=f"{host}/serving-endpoints")

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
        url = self.servers[namespace]
        auth = self._make_auth(bearer_token)

        async with FastMCPClient(url, auth=auth) as client:
            result = await client.call_tool(original_tool_name, args)
            return str(result)

mcp_registry = MultiMCPRegistry(MCP_SERVERS)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

@app.get("/")
def read_root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="index.html not found")

@app.get("/api/config")
def read_config():
    return {"model": ENDPOINT_NAME, "mcp_servers": list(MCP_SERVERS.keys())}

@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: ChatRequest, request: Request):
    try:
        user_email = request.headers.get("x-forwarded-user", "demo_user@workspace.com")

        # X-Forwarded-Access-Token is injected by Databricks Apps on every
        # authenticated request — it's the user's own OAuth token and is the
        # most reliable credential for calling other Databricks Apps.
        forwarded_token = request.headers.get("x-forwarded-access-token", "")
        bearer_token = _resolve_bearer_token(forwarded_token)
        if not bearer_token:
            logger.warning("No bearer token resolved; MCP calls will likely fail with 401")

        openai_client = get_openai_client()

        messages_payload = [
            {
                "role": "system",
                "content": (
                    f"You are Research Copilot assisting {user_email}. "
                    "You have access to tools defined in the 'tools' parameter of this request. "
                    "To retrieve database metrics, papers, notes, or reading progress, call the "
                    "appropriate function using its EXACT name and arguments from the 'tools' list. "
                    "Do not invent tool names or arguments. Wait for the function result, then answer "
                    "using that data. Never claim to have called a tool you did not actually call, "
                    "and never output text like 'tool_call' in your reply."
                )
            }
        ]
        for m in req.messages:
            messages_payload.append({"role": m.role, "content": m.content})

        agent = run_agent(user_email, messages_payload, openai_client, bearer_token)
        return StreamingResponse(_sse_encoder(agent), media_type="text/event-stream")
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


async def _sse_encoder(events):
    async for event in events:
        yield f"data: {json.dumps(event)}\n\n"


async def run_agent(user_email: str, messages_payload: list, openai_client: OpenAI, bearer_token: str = ""):
    """Orchestrate the chat agent: MCP tool discovery, LLM call, tool calls, synthesis.

    The whole flow is one MLflow AGENT trace; each MCP tool invocation is a nested
    TOOL span and each LLM call (auto-traced by mlflow.openai.autolog) is a nested
    CHAT_MODEL span. Yields SSE event dicts.
    """
    last_user_msg = messages_payload[-1]["content"] if messages_payload else ""
    events_collected = []

    def _track_event(ev):
        events_collected.append(ev)
        return ev

    try:
        try:
            agent_span_cm = mlflow.start_span(name="research_copilot_agent", span_type=SpanType.AGENT)
            agent_span = agent_span_cm.__enter__()
            agent_span.set_inputs({"user": user_email, "messages": last_user_msg})
        except Exception as mlflow_err:
            logger.warning("Could not start MLflow AGENT span: %s", mlflow_err)
            agent_span_cm = None
            agent_span = None

        yield _track_event({"type": "status", "content": "Discovering MCP tools..."})
        openai_tools, tool_router = await mcp_registry.get_all_tools(bearer_token=bearer_token)

        if agent_span is not None:
            agent_span.set_attributes({
                "mcp_tools_loaded": [t["function"]["name"] for t in openai_tools],
                "mcp_tools_count": len(openai_tools),
            })

        if openai_tools:
            tool_names = [t["function"]["name"] for t in openai_tools]
            messages_payload[0]["content"] = (
                messages_payload[0]["content"]
                + "\n\nAvailable tools (use only these exact names): " + ", ".join(tool_names)
            )

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

        yield {"type": "status", "content": "Querying model..."}

        response = openai_client.chat.completions.create(
            model=ENDPOINT_NAME,
            messages=messages_payload,
            tools=openai_tools if openai_tools else None,
            tool_choice="auto" if openai_tools else None,
            stream=False
        )

        message = response.choices[0].message
        assistant_content = message.content or ""

        tool_calls_data = {}
        if message.tool_calls:
            for idx, tc in enumerate(message.tool_calls):
                tool_calls_data[idx] = {
                    "id": tc.id,
                    "name": tc.function.name or "",
                    "args": tc.function.arguments or "",
                }

        emitted = list(tool_calls_data.values())
        if agent_span is not None:
            agent_span.set_attributes({
                "llm_emitted_tool_calls": [t["name"] for t in emitted],
                "llm_tool_calls_count": len(emitted),
            })
        if openai_tools and not emitted:
            logger.warning(
                "Tools were provided to the model but it did not emit any tool calls; "
                "it may fabricate a tool-like answer instead."
            )

        if not emitted:
            if assistant_content:
                yield {"type": "chunk", "content": assistant_content}
            yield {"type": "done"}
            return

        messages_payload.append({
            "role": "assistant",
            "content": assistant_content or None,
            "tool_calls": [
                {
                    "id": t["id"],
                    "type": "function",
                    "function": {"name": t["name"], "arguments": t["args"]}
                } for t in tool_calls_data.values()
            ]
        })

        by_original = {orig: ns_name for ns_name, (ns, orig) in tool_router.items()}
        for t in tool_calls_data.values():
            tool_name = t["name"]
            tool_args = json.loads(t["args"] or "{}")

            resolved = tool_name
            if resolved not in tool_router and tool_name in by_original:
                resolved = by_original[tool_name]
                logger.info("Resolved tool call '%s' -> '%s'", tool_name, resolved)
            if resolved not in tool_router:
                logger.error("Model called unknown tool '%s'", tool_name)
                raise ValueError(f"Unknown tool: {tool_name}")

            _, original_tool_name = tool_router[resolved]

            yield {"type": "tool_start", "name": resolved, "args": tool_args}

            with mlflow.start_span(
                name=f"tool_{original_tool_name}",
                span_type=SpanType.TOOL,
            ) as tool_span:
                # Register inputs BEFORE calling so MLflow captures them
                # even if the tool raises an exception.
                tool_span.set_inputs({
                    "tool_name": resolved,
                    "original_tool_name": original_tool_name,
                    "arguments": tool_args,
                })
                try:
                    tool_output = await mcp_registry.call_tool(
                        resolved, tool_args, tool_router, bearer_token=bearer_token
                    )
                    tool_span.set_outputs({"output": tool_output, "status": "success"})
                except Exception as tool_err:
                    error_msg = str(tool_err)
                    tool_span.set_outputs({"error": error_msg, "status": "error"})
                    logger.exception("Tool '%s' raised an exception", resolved)
                    raise

            yield {"type": "tool_end", "name": resolved, "output": tool_output}

            messages_payload.append({
                "role": "tool",
                "tool_call_id": t["id"],
                "name": tool_name,
                "content": tool_output
            })

        yield {"type": "status", "content": "Synthesizing final answer..."}

        final_stream = openai_client.chat.completions.create(
            model=ENDPOINT_NAME,
            messages=messages_payload,
            stream=True
        )

        for chunk in final_stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield {"type": "chunk", "content": delta.content}

        yield {"type": "done"}

    except Exception as e:
        logger.exception("Stream execution error")
        err_ev = {
            "type": "error",
            "content": "Something went wrong while answering.",
            "detail": str(e),
            "hint": "Retry, and check the app logs for the full traceback.",
        }
        events_collected.append(err_ev)
        yield err_ev
    finally:
        if agent_span is not None and agent_span_cm is not None:
            try:
                agent_span.set_outputs(_aggregate_agent_events(events_collected))
                agent_span_cm.__exit__(None, None, None)
            except Exception as exit_err:
                logger.warning("Could not close MLflow agent span: %s", exit_err)