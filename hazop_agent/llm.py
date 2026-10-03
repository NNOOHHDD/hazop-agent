"""Thin, provider-neutral LLM layer.

Messages use one neutral format and are converted per provider:

    {"role": "user", "content": [{"type": "text", "text": ...}, {"type": "image", "media_type": ..., "data": <base64>}]}
    {"role": "assistant", "text": ..., "tool_calls": [{"id", "name", "input"}], "raw": <provider blocks or None>}
    {"role": "tool", "results": [{"id", "name", "content": [blocks], "is_error": bool}]}

Tools are given as [{"name", "description", "input_schema"}].

Providers: anthropic (API key), openai (API key), claude_cli (local Claude Code CLI login,
optional), fake (scripted, for tests and offline smoke runs). Choose with HAZOP_LLM_PROVIDER /
HAZOP_LLM_MODEL or the --provider / --model options.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "claude_cli": "opus",
    "fake": "fake-scripted",
}


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    stop_reason: str = "end"            # end | tool_use | max_tokens | refusal | other
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    model: str = ""
    raw: Any = None                     # provider blocks, replayed verbatim on the next turn

    def as_message(self) -> dict:
        return {"role": "assistant", "text": self.text, "tool_calls": self.tool_calls, "raw": self.raw}


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def image_block(data: bytes, media_type: str = "image/png") -> dict:
    return {"type": "image", "media_type": media_type, "data": base64.standard_b64encode(data).decode("ascii")}


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in a reply (tolerates ```json fences and prose around it)."""
    if text is None:
        return None
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    for cand in fenced + [text]:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            start = cand.find(opener)
            while start != -1:
                depth, in_str, esc = 0, False, False
                for i in range(start, len(cand)):
                    ch = cand[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                    elif ch == '"':
                        in_str = True
                    elif ch == opener:
                        depth += 1
                    elif ch == closer:
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(cand[start:i + 1])
                            except (json.JSONDecodeError, ValueError):
                                break
                start = cand.find(opener, start + 1)
    return None


class LLM:
    provider = "base"

    def __init__(self, model: str):
        self.model = model

    def chat(self, system: str, messages: list[dict], tools: list[dict] | None = None,
             max_tokens: int = 16000, task: str | None = None) -> LLMResponse:
        raise NotImplementedError

    def complete(self, prompt: str, system: str = "", image: tuple[bytes, str] | None = None,
                 max_tokens: int = 16000, task: str | None = None) -> LLMResponse:
        content = [image_block(*image)] if image else []
        content.append(text_block(prompt))
        return self.chat(system, [{"role": "user", "content": content}], None, max_tokens, task)

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


# --------------------------------------------------------------------------- anthropic

class AnthropicLLM(LLM):
    """Claude via the Anthropic SDK (ANTHROPIC_API_KEY). Streams each request and returns the final message.

    HAZOP_ANTHROPIC_EFFORT   low | medium | high | xhigh | max   (default: high)
    HAZOP_ANTHROPIC_FALLBACK default | off   (server-side refusal fallback, beta; default: default)
    """
    provider = "anthropic"

    def __init__(self, model: str):
        super().__init__(model)
        import anthropic
        self.client = anthropic.Anthropic(max_retries=4)
        self.effort = os.getenv("HAZOP_ANTHROPIC_EFFORT", "high")
        self.fallback = os.getenv("HAZOP_ANTHROPIC_FALLBACK", "default").lower() != "off"

    @staticmethod
    def _blocks(content: list[dict]) -> list[dict]:
        out = []
        for b in content:
            if b["type"] == "text":
                out.append({"type": "text", "text": b["text"]})
            elif b["type"] == "image":
                out.append({"type": "image", "source": {"type": "base64", "media_type": b["media_type"], "data": b["data"]}})
        return out

    def _messages(self, messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": self._blocks(m["content"])})
            elif m["role"] == "assistant":
                if m.get("raw") is not None:
                    content = m["raw"]                      # replay thinking / tool_use blocks unchanged
                else:
                    content = ([{"type": "text", "text": m["text"]}] if m.get("text") else []) + [
                        {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["input"]} for c in m.get("tool_calls", [])]
                out.append({"role": "assistant", "content": content})
            elif m["role"] == "tool":
                out.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": r["id"], "content": self._blocks(r["content"]),
                     "is_error": bool(r.get("is_error"))} for r in m["results"]]})
        return out

    def chat(self, system, messages, tools=None, max_tokens=16000, task=None):
        kwargs: dict[str, Any] = {
            "model": self.model, "max_tokens": max_tokens, "messages": self._messages(messages),
            "output_config": {"effort": self.effort},
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"],
                                "eager_input_streaming": True} for t in tools]
        if self.fallback:
            stream = self.client.beta.messages.stream(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            stream = self.client.messages.stream(**kwargs)
        with stream as s:
            msg = s.get_final_message()
        text = "".join(b.text for b in msg.content if b.type == "text")
        calls = []
        for b in msg.content:
            if b.type == "tool_use":
                inp = b.input if isinstance(b.input, dict) else {"_unparsed_input": str(b.input)}
                calls.append({"id": b.id, "name": b.name, "input": inp})
        stop = {"end_turn": "end", "tool_use": "tool_use", "max_tokens": "max_tokens", "refusal": "refusal"}.get(msg.stop_reason, str(msg.stop_reason))
        return LLMResponse(text=text, tool_calls=calls, stop_reason=stop, model=msg.model, raw=list(msg.content),
                           usage={"input_tokens": msg.usage.input_tokens or 0, "output_tokens": msg.usage.output_tokens or 0})


# --------------------------------------------------------------------------- openai

class OpenAILLM(LLM):
    """OpenAI Chat Completions with function tools (OPENAI_API_KEY). Set the model explicitly."""
    provider = "openai"

    def __init__(self, model: str):
        super().__init__(model)
        import openai
        self.client = openai.OpenAI(max_retries=4)

    @staticmethod
    def _parts(content: list[dict]) -> list[dict]:
        out = []
        for b in content:
            if b["type"] == "text":
                out.append({"type": "text", "text": b["text"]})
            elif b["type"] == "image":
                out.append({"type": "image_url", "image_url": {"url": f"data:{b['media_type']};base64,{b['data']}"}})
        return out

    def _messages(self, system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}] if system else []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": self._parts(m["content"])})
            elif m["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("text") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                          "function": {"name": c["name"], "arguments": json.dumps(c["input"])}} for c in m["tool_calls"]]
                out.append(msg)
            elif m["role"] == "tool":
                images = []
                for r in m["results"]:
                    txt = "\n".join(b["text"] for b in r["content"] if b["type"] == "text")
                    imgs = [b for b in r["content"] if b["type"] == "image"]
                    if imgs:
                        txt += "\n(The image is attached in the next message.)"
                        images.append((r["name"], imgs))
                    out.append({"role": "tool", "tool_call_id": r["id"], "content": txt or "(no text)"})
                for name, imgs in images:   # tool messages cannot carry images in Chat Completions
                    out.append({"role": "user", "content": [{"type": "text", "text": f"Image returned by {name}:"}] + self._parts(imgs)})
        return out

    def chat(self, system, messages, tools=None, max_tokens=16000, task=None):
        kwargs: dict[str, Any] = {"model": self.model, "messages": self._messages(system, messages),
                                  "max_completion_tokens": max_tokens}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                                 "parameters": t["input_schema"]}} for t in tools]
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        calls = []
        for c in choice.message.tool_calls or []:
            try:
                inp = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                inp = {"_unparsed_input": c.function.arguments}
            calls.append({"id": c.id, "name": c.function.name, "input": inp if isinstance(inp, dict) else {"_unparsed_input": inp}})
        stop = {"stop": "end", "tool_calls": "tool_use", "length": "max_tokens", "content_filter": "refusal"}.get(choice.finish_reason, str(choice.finish_reason))
        usage = {"input_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0, "output_tokens": getattr(resp.usage, "completion_tokens", 0) or 0}
        return LLMResponse(text=choice.message.content or "", tool_calls=calls, stop_reason=stop, usage=usage, model=resp.model)


# --------------------------------------------------------------------------- claude_cli (optional)

CLI_SYSTEM_PROMPT = "You are a careful engineering assistant. Follow the user's instructions exactly."


class ClaudeCLILLM(LLM):
    """Local Claude Code CLI (`claude -p`) under the user's own subscription login. Optional.

    The CLI cannot receive custom tool definitions, so tool use is emulated with a JSON reply
    protocol: every call re-sends the whole transcript, and images are written to files that the
    model opens with the CLI's Read tool. Slower than the API providers; meant for trying the
    agent without an API key.
    """
    provider = "claude_cli"

    def __init__(self, model: str):
        super().__init__(model)
        self.exe = os.getenv("CLAUDE_CLI_PATH") or shutil.which("claude")
        if not self.exe:
            raise RuntimeError("claude CLI not found on PATH (set CLAUDE_CLI_PATH).")
        self.workdir = Path(os.getenv("HAZOP_CLI_WORKDIR") or tempfile.mkdtemp(prefix="hazop_cli_"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.timeout = float(os.getenv("HAZOP_CLI_TIMEOUT", "1800"))
        self._n = 0

    def _image_file(self, b: dict) -> str:
        data = base64.standard_b64decode(b["data"])
        ext = ".png" if "png" in b["media_type"] else ".jpg"
        p = self.workdir / (hashlib.sha256(data).hexdigest()[:16] + ext)
        if not p.exists():
            p.write_bytes(data)
        return str(p.resolve())

    def _render(self, content: list[dict]) -> tuple[str, bool]:
        parts, has_img = [], False
        for b in content:
            if b["type"] == "text":
                parts.append(b["text"])
            elif b["type"] == "image":
                has_img = True
                parts.append(f"[Image file: {self._image_file(b)} - open it with the Read tool]")
        return "\n".join(parts), has_img

    def build_prompt(self, system: str, messages: list[dict], tools: list[dict] | None) -> tuple[str, bool]:
        lines, any_img = [], False
        if system:
            lines += ["# Instructions", system, ""]
        if tools:
            lines += ["# Tools", "You can call these tools (JSON Schema of each input):"]
            lines += [json.dumps({"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}) for t in tools]
            lines += ["", "# Reply protocol",
                      'Reply with ONE JSON object and nothing else: {"text": "<optional short note>", '
                      '"tool_calls": [{"name": "<tool>", "input": {...}}]}. You may put several calls in the list. '
                      'The results come back in the next turn. Use an empty list only when you have nothing more to do.', ""]
        lines.append("# Conversation")
        for m in messages:
            if m["role"] == "user":
                txt, img = self._render(m["content"])
                any_img |= img
                lines += ["## USER", txt]
            elif m["role"] == "assistant":
                lines += ["## ASSISTANT", json.dumps({"text": m.get("text", ""),
                                                      "tool_calls": [{"name": c["name"], "input": c["input"]} for c in m.get("tool_calls", [])]})]
            elif m["role"] == "tool":
                for r in m["results"]:
                    txt, img = self._render(r["content"])
                    any_img |= img
                    lines += [f"## TOOL RESULT ({r['name']}{', error' if r.get('is_error') else ''})", txt]
        lines += ["", "## ASSISTANT (your reply)"]
        return "\n".join(lines), any_img

    def chat(self, system, messages, tools=None, max_tokens=16000, task=None):
        prompt, has_img = self.build_prompt(system, messages, tools)
        tool_flags = ["--tools", "Read", "--allowedTools", "Read"] if has_img else ["--tools", ""]
        cmd = [self.exe, "-p", "--output-format", "json", "--model", self.model, "--max-turns", "6",
               "--system-prompt", CLI_SYSTEM_PROMPT] + tool_flags
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)       # allow launching from inside a Claude Code terminal
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", env=env, cwd=self.workdir, timeout=self.timeout)
        data = extract_json(proc.stdout or "")
        if not isinstance(data, dict) or "result" not in data:
            raise RuntimeError(f"claude CLI failed (exit {proc.returncode}): {(proc.stderr or proc.stdout)[-400:]}")
        if data.get("is_error"):
            raise RuntimeError(f"claude CLI error: {str(data.get('result'))[:300]}")
        return self.parse_reply(str(data.get("result") or ""), tools, data)

    def parse_reply(self, result: str, tools, data: dict | None = None) -> LLMResponse:
        u = (data or {}).get("usage") or {}
        usage = {"input_tokens": int(u.get("input_tokens") or 0) + int(u.get("cache_read_input_tokens") or 0)
                 + int(u.get("cache_creation_input_tokens") or 0),
                 "output_tokens": int(u.get("output_tokens") or 0)}
        if not tools:
            return LLMResponse(text=result, usage=usage, model=self.model)
        obj = extract_json(result)
        if not isinstance(obj, dict):
            return LLMResponse(text=result, usage=usage, model=self.model)
        calls = []
        for c in obj.get("tool_calls") or []:
            if isinstance(c, dict) and c.get("name"):
                self._n += 1
                inp = c.get("input") if isinstance(c.get("input"), dict) else {}
                calls.append({"id": f"cli_{self._n}", "name": str(c["name"]), "input": inp})
        return LLMResponse(text=str(obj.get("text") or ""), tool_calls=calls,
                           stop_reason="tool_use" if calls else "end", usage=usage, model=self.model)


# --------------------------------------------------------------------------- factory

def get_llm(provider: str | None = None, model: str | None = None) -> LLM:
    env_provider = (os.getenv("HAZOP_LLM_PROVIDER") or "anthropic").strip().lower()
    provider = (provider or env_provider).strip().lower()
    # HAZOP_LLM_MODEL belongs to HAZOP_LLM_PROVIDER; another provider falls back to its own default
    model = model or (os.getenv("HAZOP_LLM_MODEL") if provider == env_provider else None) or DEFAULT_MODELS.get(provider)
    if provider == "fake":
        from .fake import FakeLLM
        return FakeLLM()
    if not model:
        raise SystemExit(f"Set a model for provider '{provider}' (HAZOP_LLM_MODEL or --model).")
    if provider == "anthropic":
        return AnthropicLLM(model)
    if provider == "openai":
        return OpenAILLM(model)
    if provider == "claude_cli":
        return ClaudeCLILLM(model)
    raise SystemExit(f"Unknown provider '{provider}'. Use anthropic, openai, claude_cli or fake.")
