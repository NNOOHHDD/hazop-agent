import asyncio

import pytest

pytest.importorskip("mcp")


def test_mcp_server_exposes_two_tools():
    from hazop_agent.mcp_server import server

    tools = asyncio.run(server.list_tools())
    assert sorted(t.name for t in tools) == ["evaluate_worksheet", "generate_worksheet"]
