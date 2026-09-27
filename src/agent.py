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
        #   "sarvam-m" / "sarvam/…"  -> Sarvam AI hosted API (challenge baseline;
        #                               India-hosted, declared in submission)
        #   anything else            -> Ollama
        if model.startswith("Qwen/"):
            self.backend = "vllm"
            self.model = model
        elif model == "hosted":
            # Generic India-hosted API baseline (Krutrim / Sarvam / Azure-India).
            # All config via .env: API_BASE_URL, API_KEY, API_MODEL
            self.backend = "hosted"
            self.model = os.environ.get("API_MODEL", "krutrim-1")
        elif model.startswith("sarvam"):
            self.backend = "sarvam"
            # "sarvam/xyz" -> "xyz"; bare "sarvam-m" stays as-is
            self.model = model.split("/", 1)[1] if "/" in model else model
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
            "think":    False,  # TOP-LEVEL parameter — required. Placing it inside
                                 # `options` is silently ignored by Ollama and leaves
                                 # thinking mode ON, which invalidates latency figures
                                 # (challenge spec §5).
            "options":  {"temperature": 0.3, "num_predict": 1024, "num_ctx": 8192},
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

    def _call_sarvam(self, messages: list[dict], use_tools: bool = True) -> tuple[dict, float]:
        """Sarvam AI hosted chat completions (OpenAI-compatible shape).
        Config via env: SARVAM_API_KEY (required), SARVAM_BASE_URL, SARVAM_MODEL.
        If the endpoint rejects the `tools` parameter, retries without tools and
        relies on text-level tool-call parsing downstream."""
        api_key = os.environ.get("SARVAM_API_KEY")
        if not api_key:
            raise RuntimeError("SARVAM_API_KEY not set (add it to .env)")
        base_url = os.environ.get("SARVAM_BASE_URL", "https://api.sarvam.ai")
        model    = os.environ.get("SARVAM_MODEL", self.model)
        headers  = {
            "Authorization": f"Bearer {api_key}",
            "api-subscription-key": api_key,  # Sarvam accepts either header style
            "Content-Type": "application/json",
        }
        payload = {
            "model":       model,
            "messages":    messages,
            "temperature": 0.3,
            "max_tokens":  1024,
        }
        if use_tools:
            payload["tools"] = _tool_schemas()

        def _post(p: dict) -> tuple[dict, float]:
            """POST with rate-limit hardening: honour Retry-After on 429,
            exponential backoff on 5xx/timeouts, 6 attempts max."""
            t0 = time.perf_counter()
            backoff = 5.0
            last_err = None
            for attempt in range(6):
                try:
                    r = requests.post(f"{base_url}/v1/chat/completions",
                                      json=p, headers=headers, timeout=120)
                    if r.status_code == 429:
                        wait = float(r.headers.get("Retry-After") or backoff)
                        print(f"    [rate-limit] 429 — waiting {wait:.0f}s "
                              f"(attempt {attempt+1}/6)")
                        time.sleep(wait)
                        backoff = min(backoff * 2, 120)
                        last_err = "HTTP 429"
                        continue
                    if r.status_code >= 500:
                        print(f"    [retry] HTTP {r.status_code} — backoff {backoff:.0f}s")
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 120)
                        last_err = f"HTTP {r.status_code}"
                        continue
                    dt_try = time.perf_counter() - t0
                    r.raise_for_status()
                    return r.json(), dt_try
                except (requests.Timeout, requests.ConnectionError) as e:
                    print(f"    [retry] {type(e).__name__} — backoff {backoff:.0f}s")
                    last_err = type(e).__name__
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 120)
            raise RuntimeError(f"Sarvam API failed after 6 attempts: {last_err}")

        try:
            data, dt = _post(payload)
        except requests.HTTPError as e:
            resp = e.response
            if use_tools and resp is not None and resp.status_code < 500 \
                    and "tool" in (resp.text or "").lower():
                print("    endpoint rejected tools param — retrying without tools")
                payload.pop("tools", None)
                data, dt = _post(payload)
            else:
                raise
        msg = data["choices"][0]["message"]
        # normalise OpenAI-style tool_calls into our internal shape
        if msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            fn = tc.get("function", tc)
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            msg["tool_calls"] = [{"function": {"name": fn.get("name"), "arguments": args}}]
        return msg, dt

    def _call_hosted(self, messages: list[dict], use_tools: bool = True) -> tuple[dict, float]:
        """Generic OpenAI-compatible hosted endpoint, config from env:
        API_BASE_URL (e.g. https://api.olakrutrim.com/<region>/v1), API_KEY, API_MODEL.
        Same rate-limit hardening as the Sarvam client."""
        api_key  = os.environ.get("API_KEY")
        base_url = os.environ.get("API_BASE_URL", "").rstrip("/")
        if not api_key or not base_url:
            raise RuntimeError("API_KEY / API_BASE_URL not set (add them to .env)")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model":       self.model,
            "messages":    messages,
            "temperature": 0.3,
            "max_tokens":  1024,
        }
        if use_tools:
            payload["tools"] = _tool_schemas()

        t0 = time.perf_counter()
        backoff = 5.0
        last_err = None
        for attempt in range(6):
            try:
                r = requests.post(f"{base_url}/chat/completions",
                                  json=payload, headers=headers, timeout=120)
                if r.status_code == 429:
                    wait = float(r.headers.get("Retry-After") or backoff)
                    print(f"    [rate-limit] 429 — waiting {wait:.0f}s (attempt {attempt+1}/6)")
                    time.sleep(wait)
                    backoff = min(backoff * 2, 120)
                    last_err = "HTTP 429"
                    continue
                if r.status_code >= 500:
                    print(f"    [retry] HTTP {r.status_code} — backoff {backoff:.0f}s")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 120)
                    last_err = f"HTTP {r.status_code}"
                    continue
                dt = time.perf_counter() - t0
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                if msg.get("tool_calls"):
                    tc = msg["tool_calls"][0]
                    fn = tc.get("function", tc)
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    msg["tool_calls"] = [{"function": {"name": fn.get("name"), "arguments": args}}]
                return msg, dt
            except (requests.Timeout, requests.ConnectionError) as e:
                print(f"    [retry] {type(e).__name__} — backoff {backoff:.0f}s")
                last_err = type(e).__name__
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
        raise RuntimeError(f"hosted API failed after 6 attempts: {last_err}")

    def turn(self, borrower_text: str, use_tools: bool = True) -> dict:
        self.history.append({"role": "user", "content": borrower_text})
        messages = [{"role": "system", "content": self.system}] + self.history

        if self.backend == "ollama":
            msg, ttft = self._call_ollama(messages, use_tools)
        elif self.backend == "sarvam":
            msg, ttft = self._call_sarvam(messages, use_tools)
        elif self.backend == "hosted":
            msg, ttft = self._call_hosted(messages, use_tools)
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
