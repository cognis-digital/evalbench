"""EVALBENCH MCP server — exposes evaluate_suite() as an MCP tool for Cognis.Studio."""
from __future__ import annotations
import json
from evalbench.core import evaluate_suite, EvalError


def serve() -> int:
    """Start an MCP stdio server. Requires the optional 'mcp' extra:
        pip install "cognis-evalbench[mcp]"
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception:
        print("Install the MCP extra: pip install 'cognis-evalbench[mcp]'")
        return 1
    app = FastMCP("evalbench")

    @app.tool()
    def evalbench_run(suite_json: str) -> str:
        """Evaluate a suite JSON string with the evalbench harness. Returns JSON run result."""
        try:
            suite = json.loads(suite_json)
        except (ValueError, TypeError) as exc:
            return json.dumps({"error": f"invalid JSON: {exc}"})
        try:
            run = evaluate_suite(suite)
        except EvalError as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps(run.to_dict(), indent=2)

    app.run()
    return 0
