"""
run_chain_today.py — end-to-end chain for 2026-09-27.

Context: PS-1 for qwen-voice and sarvam/sarvam-105b-conversations is already
complete. A PS-1 re-run for qwen3.5:4b is in flight (writes
results/ps1_qwen3.5_4b.jsonl). This script:

  1. waits for the 4b re-run to finish (file quiet + 192 valid rows),
  2. rebuilds the PS-1 summary across ALL THREE models,
  3. runs PS-2 then PS-3 for all three models (sequential, resume-safe),
  4. regenerates combined_results.json, report/findings.md and charts.

Launch detached:
    cd src && python run_chain_today.py
Resume-safe: every stage checks its own outputs before running.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv  # noqa: F401  (agent.py loads .env itself)

from ps1_guardrail import _read_rows, RESULTS_DIR, JUDGE_VERSION, \
    compute_summary as ps1_summary, save_results as ps1_save
from ps2_register import run_ps2, save_results as save_ps2
from ps3_tool_calls import run_ps3, compute_ps3_summary, save_ps3_results

LOCAL_MODELS = ["qwen3.5:4b", "qwen-voice"]
API_MODELS   = ["sarvam/sarvam-105b-conversations"]
ALL_MODELS   = LOCAL_MODELS + API_MODELS
PS1_CASES    = 192
CHAIN_LOG    = RESULTS_DIR / "chain_log.txt"
STABLE_SECONDS = 120   # JSONL must be untouched this long before we call a run done


def _safe(m: str) -> str:
    return m.replace(":", "_").replace("/", "_")


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(CHAIN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _file_quiet(path: Path) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) > STABLE_SECONDS


def ps1_4b_complete() -> bool:
    out = RESULTS_DIR / "ps1_qwen3.5_4b.jsonl"
    if not out.exists() or not _file_quiet(out):
        return False
    rows = [r for r in _read_rows(out)
            if r.get("judge_version") == JUDGE_VERSION and r.get("violated") is not None]
    return len({r["id"] for r in rows}) >= PS1_CASES


def wait_for_4b():
    log(f"Waiting for qwen3.5:4b PS-1 re-run ({PS1_CASES} cases, file quiet >{STABLE_SECONDS}s)...")
    while not ps1_4b_complete():
        out = RESULTS_DIR / "ps1_qwen3.5_4b.jsonl"
        n = len(_read_rows(out)) if out.exists() else 0
        log(f"waiting: qwen3.5:4b={n}")
        time.sleep(300)
    log("qwen3.5:4b PS-1 complete.")


def stage_ps1_summary():
    """Rebuild one PS-1 summary covering all three models from the JSONLs."""
    log("Rebuilding combined PS-1 summary for all models...")
    case_ids = set()
    data = RESULTS_DIR.parent / "data" / "test_cases.jsonl"
    if data.exists():
        case_ids = {json.loads(l)["id"] for l in data.read_text(encoding="utf-8").splitlines()
                    if l.strip()}
    all_results = {}
    for m in ALL_MODELS:
        out = RESULTS_DIR / f"ps1_{_safe(m)}.jsonl"
        rows = _read_rows(out) if out.exists() else []
        if case_ids:
            rows = [r for r in rows if r["id"] in case_ids]
        all_results[m] = rows
    s = ps1_summary(all_results)
    ps1_save(all_results, s)
    log("PS-1 summary rebuilt (ps1_summary.json covers all 3 models).")


def stage_ps2():
    todo = [m for m in ALL_MODELS
            if not (RESULTS_DIR / f"ps2_{_safe(m)}.jsonl").exists()]
    if not todo:
        log("PS-2 outputs already present — skipping.")
        return
    log(f"PS-2 starting for {todo} (register + TTS round-trip)...")
    try:
        r = run_ps2(todo)
        save_ps2(r)
        log("PS-2 done.")
    except Exception as e:
        log(f"ERROR in PS-2: {e} — continuing chain; rerun later to fill gap.")


def stage_ps3():
    todo = [m for m in ALL_MODELS
            if not (RESULTS_DIR / f"ps3_{_safe(m)}.jsonl").exists()]
    if not todo:
        log("PS-3 outputs already present — skipping.")
        return
    log(f"PS-3 starting for {todo} (tool calls under code-mixing). "
        "This is the longest stage (~100 min per local model on CPU)...")
    try:
        r = run_ps3(todo)
        s = compute_ps3_summary(r)
        save_ps3_results(r, s)
        log("PS-3 done.")
    except Exception as e:
        log(f"ERROR in PS-3: {e} — continuing chain; rerun later to fill gap.")


def stage_finalise():
    log("Rebuilding combined_results.json + findings.md + charts...")
    combined = {"models": ALL_MODELS, "results": {}}

    ps1_path = RESULTS_DIR / "ps1_summary.json"
    if ps1_path.exists():
        combined["results"]["ps1"] = json.loads(ps1_path.read_text(encoding="utf-8"))
        combined["results"]["ps1_judge_version"] = JUDGE_VERSION

    ps2 = {}
    for m in ALL_MODELS:
        p = RESULTS_DIR / f"ps2_{_safe(m)}.jsonl"
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

    subprocess.run([sys.executable, "analysis/generate_charts.py"],
                   cwd=RESULTS_DIR.parent, check=False)

    log("Finalisation done.")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    log("=== chain started ===")
    wait_for_4b()
    stage_ps1_summary()
    stage_ps2()
    stage_ps3()
    stage_finalise()
    log("CHAIN COMPLETE — Track 1 end-to-end done for all 3 models.")
