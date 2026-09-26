"""
Core collections agent — supports Ollama (Track 1), vLLM (Track 2).
Pass model as:
  "qwen3.5:4b"       -> Ollama
  "qwen-voice"       -> Ollama
  "Qwen/..."         -> vLLM
"""
import json
import os
import time
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

SCHEMAS_PATH = Path(__file__).parent.parent / "schemas" / "tools.json"
PROMPT_PATH  = Path(__file__).parent.parent / "prompts" / "baseline.txt"

TOOLS: list[dict] = json.loads(SCHEMAS_PATH.read_text())
BASE_PROMPT: str  = PROMPT_PATH.read_text()

OLLAMA_URL = "http://localhost:11434/api/chat"
VLLM_URL   = "http://localhost:8000/v1/chat/completions"


def build_system_prompt(lender="PredixionBank", name="Borrower",
                        dpd=30, product="Personal Loan", amount=10000) -> str:
    return (BASE_PROMPT
            .replace("{LENDER}", lender)
            .replace("{NAME}", name)
            .replace("{DPD}", str(dpd))
            .replace("{PRODUCT}", product)
            .replace("{AMOUNT}", str(amount)))


def _tool_schemas() -> list[dict]:
    return [{"type": "function", "function": t} for t in TOOLS]


def _parse_tool_call(content: str) -> dict | None:
    """Extract a tool call from text content. Handles nested JSON arguments."""
    try:
        start = content.find('{')
        while start != -1:
            end = content.rfind('}', start)
            if end == -1:
                break
            candidate = content[start:end + 1]
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict) and "name" in obj:
                    return obj
            except json.JSONDecodeError:
                pass
            start = content.find('{', start + 1)
    except Exception:
        pass
    return None


def _extract_tool(msg: dict) -> dict | None:
    if msg.get("tool_calls"):
        tc = msg["tool_calls"][0]
        fn = tc.get("function", tc)
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        return {"name": fn.get("name"), "arguments": args}
    if msg.get("content"):
        return _parse_tool_call(msg["content"])
    return None


def _tool_reply_fallback(tool_call: dict) -> str:
    """Generate a short text reply when the model emits only a tool call (no text).
    Used so PS-2 TTS evaluation always has audio content."""
    name = (tool_call.get("name") or "").lower()
    if "dispute" in name:
        return "मैं समझ रहा हूँ। आपका मामला देख रहा हूँ।"
    if "escalate" in name:
        return "एक मानव एजेंट को ट्रांसफर कर रहा हूँ।"
    if "disposition" in name:
        return "धन्यवाद। आपका अगला कॉल हम फॉलो अप करेंगे।"
    if "capture" in name:
        return "आपका भुगतान का वादा दर्ज कर रहा हूँ।"
    if "payment_link" in name:
        return "भुगतान लिंक भेज रहा हूँ।"
    return "हम आपकी तकनीकी सहायता प्रदान करने के लिए यहाँ हैं।"


class CollectionsAgent:
    def __init__(self, model: str = "qwen-voice",
                 lender="PredixionBank", name="Borrower",
                 dpd=30, product="Personal Loan", amount=10000):
        # model formats:
        #   "Qwen/..."               -> vLLM
        #   anything else            -> Ollama
        if model.startswith("Qwen/"):
            self.backend = "vllm"
            self.model = model
        else:
            self.backend = "ollama"
            self.model = model

        self.system  = build_system_prompt(lender, name, dpd, product, amount)
        self.history: list[dict] = []
        self.tool_calls_made: list[dict] = []

    def _call_ollama(self, messages: list[dict], use_tools: bool = True) -> tuple[dict, float]:
        # 9b model needs more time; 4b is faster
        is_9b = "9b" in self.model.lower() or "qwen-voice" in self.model.lower()
        timeout = 1800 if is_9b else 600  # 30 min for 9b, 10 min for 4b
        payload = {
            "model":    self.model,
            "messages": messages,
            "tools":    _tool_schemas() if use_tools else [],
            "stream":   False,
            "options":  {"temperature": 0.3, "num_predict": 4096, "num_ctx": 8192, "think": False},
        }
        t0 = time.perf_counter()
        r  = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["message"], time.perf_counter() - t0

    def _call_vllm(self, messages: list[dict]) -> tuple[dict, float]:
        payload = {
            "model":       self.model,
            "messages":    messages,
            "tools":       _tool_schemas(),
            "tool_choice": "auto",
            "temperature": 0.3,
            "max_tokens":  4096,
        }
        t0 = time.perf_counter()
        r  = requests.post(VLLM_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"], time.perf_counter() - t0

    def turn(self, borrower_text: str, use_tools: bool = True) -> dict:
        self.history.append({"role": "user", "content": borrower_text})
        messages = [{"role": "system", "content": self.system}] + self.history

        if self.backend == "ollama":
            msg, ttft = self._call_ollama(messages, use_tools)
        else:
            msg, ttft = self._call_vllm(messages)

        tool_call = _extract_tool(msg)
        reply     = (msg.get("content") or "").strip()

        if not reply and tool_call:
            reply = _tool_reply_fallback(tool_call)

        self.history.append({"role": "assistant", "content": reply})
        if tool_call:
            self.tool_calls_made.append(tool_call)

        return {"reply": reply, "tool_call": tool_call, "ttft": round(ttft, 3)}

    def reset(self):
        self.history.clear()
        self.tool_calls_made.clear()
