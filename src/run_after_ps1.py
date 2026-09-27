"""
Watcher chain: waits for the local PS-1 run (both Ollama models) to complete,
then automatically:
  1. runs PS-1 for the Sarvam hosted baseline (same suite, same judge),
  2. runs PS-2 and PS-3 for all three models,
  3. refreshes combined_results.json + report/findings.md.

Launch detached:  python run_after_ps1.py
Resume-safe: every stage checks its own outputs before running.
Requires SARVAM_API_KEY in .env by the time stage 1 starts.
"""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from ps1_guardrail import _read_rows, RESULTS_DIR, JUDGE_VERSION, run_ps1, \
    compute_summary as ps1_summary, save_results as ps1_save
from ps2_register import run_ps2, save_results as save_ps2
from ps3_tool_calls import run_ps3, compute_ps3_summary, save_ps3_results

load_dotenv(Path(__file__).parent.parent / ".env")

LOCAL_MODELS = ["qwen3.5:4b", "qwen-voice"]
API_MODELS   = ["hosted"]   # generic India-hosted baseline; config via .env:
                            # API_BASE_URL, API_KEY, API_MODEL (Krutrim / Sarvam / Azure-India)
ALL_MODELS   = LOCAL_MODELS + API_MODELS
PS1_CASES    = 192
CHAIN_LOG    = RESULTS_DIR / "chain_log.txt"
STABLE_SECONDS = 90   # file must be untouched this long before we consider it done


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(CHAIN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _file_quiet(path: Path) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) > STABLE_SECONDS


def ps1_complete() -> bool:
    for model in LOCAL_MODELS:
        safe = model.replace(":", "_").replace("/", "_")
        out = RESULTS_DIR / f"ps1_{safe}.jsonl"
        if not out.exists():
            return False
        rows = [r for r in _read_rows(out)
                if r.get("judge_version") == JUDGE_VERSION and r.get("violated") is not None]
        if len({r["id"] for r in rows}) < PS1_CASES:
            return False
        if not _file_quiet(out):
            return False
    return True


def wait_for_ps1():
    log(f"Watching for local PS-1 completion ({PS1_CASES} cases x {len(LOCAL_MODELS)} models)...")
    while not ps1_complete():
        counts = []
        for model in LOCAL_MODELS:
            safe = model.replace(":", "_").replace("/", "_")
            out = RESULTS_DIR / f"ps1_{safe}.jsonl"
            n = len(_read_rows(out)) if out.exists() else 0
            counts.append(f"{model}={n}")
        log("waiting: " + " ".join(counts))
        time.sleep(600)
    log("Local PS-1 complete — starting chain.")


def stage_ps1_api():
    safe = API_MODELS[0].replace(":", "_").replace("/", "_")
    out = RESULTS_DIR / f"ps1_{safe}.jsonl"
    if out.exists():
        rows = [r for r in _read_rows(out) if r.get("violated") is not None]
        if len({r["id"] for r in rows}) >= PS1_CASES:
            log("Sarvam PS-1 already complete — skipping.")
            return
    if not os.environ.get("API_KEY") or not os.environ.get("API_BASE_URL"):
        log("ERROR: API_KEY / API_BASE_URL not set — hosted baseline stage skipped. "
            "Add them to .env and rerun this script; it will resume.")
        return
    log("PS-1 starting for hosted API baseline (192 cases, same judge)...")
    try:
        r = run_ps1(API_MODELS + LOCAL_MODELS, judge_model="qwen-voice")
        # passing LOCAL_MODELS too: their rows are complete, so this only
        # rebuilds a single summary covering all three models
        s = ps1_summary(r)
        ps1_save(r, s)
        log("Sarvam PS-1 done (summary now covers all models).")
    except Exception as e:
        log(f"ERROR in Sarvam PS-1: {e} — continuing chain; rerun later to fill gap.")


def stage_ps2():
    todo = [m for m in ALL_MODELS
            if not (RESULTS_DIR / f"ps2_{m.replace(':', '_').replace('/', '_')}.jsonl").exists()]
    if not todo:
        log("PS-2 outputs already present — skipping.")
        return
    log(f"PS-2 starting for {todo} (register + TTS round-trip)...")
    r = run_ps2(todo)
    save_ps2(r)
    log("PS-2 done.")


def stage_ps3():
    todo = [m for m in ALL_MODELS
            if not (RESULTS_DIR / f"ps3_{m.replace(':', '_').replace('/', '_')}.jsonl").exists()]
    if not todo:
        log("PS-3 outputs already present — skipping.")
        return
    log(f"PS-3 starting for {todo} (tool calls under code-mixing)...")
    r = run_ps3(todo)
    s = compute_ps3_summary(r)
    save_ps3_results(r, s)
    log("PS-3 done.")


def stage_finalise():
    log("Rebuilding combined_results.json + findings.md ...")
    combined = {"models": ALL_MODELS, "results": {}}

    ps1_path = RESULTS_DIR / "ps1_summary.json"
    if ps1_path.exists():
        combined["results"]["ps1"] = json.loads(ps1_path.read_text(encoding="utf-8"))
        combined["results"]["ps1_judge_version"] = JUDGE_VERSION

    ps2 = {}
    for m in ALL_MODELS:
        safe = m.replace(":", "_").replace("/", "_")
        p = RESULTS_DIR / f"ps2_{safe}.jsonl"
        if p.exists():
            ps2[m] = [r for r in _read_rows(p) if "naturalness" in r]
    combined["results"]["ps2"] = ps2

    ps3_path = RESULTS_DIR / "ps3_summary.json"
    if ps3_path.exists():
        combined["results"]["ps3"] = json.loads(ps3_path.read_text(encoding="utf-8"))

    (RESULTS_DIR / "combined_results.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")

    from run_all import _write_report
    _write_report(combined, None)
    log("Finalisation done.")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    wait_for_ps1()
    stage_ps1_api()
    stage_ps2()      # sequential — shares Ollama/API cleanly
    stage_ps3()
    stage_finalise()
    log("CHAIN COMPLETE — Track 1 text tasks done for all 3 models.")
