import json

from hazop_agent.agent import run_agent, run_single, save_run
from hazop_agent.fake import FakeLLM
from hazop_agent.llm import AnthropicLLM, ClaudeCLILLM, OpenAILLM, extract_json, image_block, text_block
from hazop_agent.tools import TOOL_SPECS


def test_agent_loop_with_fake_llm(pid_path, node_path, tmp_path):
    res = run_agent(FakeLLM(), pid_path, node_path)
    meta = res["meta"]
    assert meta["status"] == "finished" and meta["rows"] >= 1 and meta["rejected_rows"] == 1
    assert all(r["node"] == "MCFC stack" for r in res["rows"])
    names = [c["name"] for s in res["log"]["steps"] for c in s["tool_calls"]]
    assert names[:3] == ["view_pid", "get_node_description", "list_guidewords"] and names[-1] == "finish"
    # images are not written into the log
    assert "base64" not in json.dumps(res["log"]) and "[image image/png]" in json.dumps(res["log"])
    out = save_run(res, tmp_path / "run_01")
    assert {p.name for p in out.iterdir()} == {"worksheet.json", "worksheet.csv", "log.json"}


def test_agent_stops_at_max_steps(pid_path, node_path):
    res = run_agent(FakeLLM(), pid_path, node_path, max_steps=2)
    assert res["meta"]["status"] == "max_steps" and res["meta"]["steps"] == 2


def test_single_mode_with_fake_llm(pid_path, node_path):
    res = run_single(FakeLLM(), pid_path, node_path)
    assert res["meta"]["mode"] == "single" and res["meta"]["status"] == "finished" and res["rows"]


def test_tool_names():
    assert [t["name"] for t in TOOL_SPECS] == ["view_pid", "get_node_description", "list_guidewords", "add_row", "finish"]


def test_extract_json():
    assert extract_json('Sure:\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert extract_json('text {"b": "x}y"} more') == {"b": "x}y"}
    assert extract_json("[1, 2]") == [1, 2]
    assert extract_json("nothing") is None


def _conversation():
    return [
        {"role": "user", "content": [text_block("go")]},
        {"role": "assistant", "text": "", "tool_calls": [{"id": "t1", "name": "view_pid", "input": {}}], "raw": None},
        {"role": "tool", "results": [{"id": "t1", "name": "view_pid", "content": [text_block("P&ID:"), image_block(b"\x89PNG")],
                                      "is_error": False}]},
    ]


def test_anthropic_message_conversion(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    msgs = AnthropicLLM("claude-opus-5-5")._messages(_conversation())
    assert msgs[1]["content"][0]["type"] == "tool_use"
    tr = msgs[2]["content"][0]
    assert msgs[2]["role"] == "user" and tr["type"] == "tool_result" and tr["tool_use_id"] == "t1"
    assert tr["content"][1]["type"] == "image" and tr["content"][1]["source"]["type"] == "base64"


def test_openai_message_conversion(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    msgs = OpenAILLM("some-model")._messages("sys", _conversation())
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "tool", "user"]
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "view_pid"
    assert msgs[4]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_claude_cli_protocol(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_CLI_PATH", "/bin/true")
    monkeypatch.setenv("HAZOP_CLI_WORKDIR", str(tmp_path))
    cli = ClaudeCLILLM("opus")
    prompt, has_img = cli.build_prompt("system text", _conversation(), TOOL_SPECS)
    assert has_img and "Read tool" in prompt and '"name": "add_row"' in prompt
    assert list(tmp_path.glob("*.png"))
    r = cli.parse_reply('{"text": "ok", "tool_calls": [{"name": "finish", "input": {"summary": "s"}}]}', TOOL_SPECS,
                        {"usage": {"input_tokens": 10, "output_tokens": 5}})
    assert r.stop_reason == "tool_use" and r.tool_calls[0]["name"] == "finish" and r.usage["output_tokens"] == 5


def test_model_env_only_applies_to_its_provider(monkeypatch):
    from hazop_agent.llm import get_llm

    monkeypatch.setenv("HAZOP_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("HAZOP_LLM_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert get_llm().model == "claude-sonnet-5-5"
    assert get_llm("fake").provider == "fake"
    try:
        get_llm("openai")
    except SystemExit as e:
        assert "Set a model" in str(e)
    else:
        raise AssertionError("openai without a model must stop")
