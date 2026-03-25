"""EVALBENCH MCP server — exposes scan() as an MCP tool for Cognis.Studio."""
from __future__ import annotations
from evalbench.core import scan, to_json

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
    def evalbench_scan(target: str) -> str:
        """Offline LLM / agent eval harness with regression gates. Returns JSON findings."""
        return to_json(scan(target))

    app.run()
    return 0
