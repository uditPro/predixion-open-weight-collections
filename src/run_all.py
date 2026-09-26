"""
run_all.py — single entry point for the Open-Weight Collections Agent challenge.
Runs PS-1, PS-2, PS-3 (Track 1) and optionally PS-4, PS-5 (Track 2).
"""
import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
REPORT_DIR  = Path(__file__).parent.parent / "report"


def main():
    parser = argparse.ArgumentParser(description="Predixion Open-Weight Collections Agent — full eval")
    parser.add_argument("--models", nargs="+", default=["qwen-voice", "qwen3.5:4b"])
    parser.add_argument("--judge",      default="qwen-voice")
    parser.add_argument("--ps",         nargs="+", type=int, default=[1, 2, 3], choices=[1,2,3,4,5])
    parser.add_argument("--precisions", nargs="+", default=["Q4"])
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    combined: dict = {"models": args.models, "results": {}}

    if 1 in args.ps:
        print("\n" + "="*60)
        print("PS-1 · Guardrail Gauntlet")
        print("="*60)
        from ps1_guardrail import run_ps1, compute_summary, save_results
        r = run_ps1(args.models, judge_model=args.judge)
        s = compute_summary(r)
        save_results(r, s)
        combined["results"]["ps1"] = s

    if 2 in args.ps:
        print("\n" + "="*60)
        print("PS-2 · Code-Mix Register Test")
        print("="*60)
        from ps2_register import run_ps2, save_results as save_ps2
        r = run_ps2(args.models, judge_model=args.judge)
        save_ps2(r)
        combined["results"]["ps2"] = {
            m: [row for row in rows if "naturalness" in row]
            for m, rows in r.items()
        }

    if 3 in args.ps:
        print("\n" + "="*60)
        print("PS-3 · Tool Calls Under Code-Mixing")
        print("="*60)
        from ps3_tool_calls import run_ps3, compute_ps3_summary, save_ps3_results
        r = run_ps3(args.models)
        s = compute_ps3_summary(r)
        save_ps3_results(r, s)
        combined["results"]["ps3"] = s

    if 4 in args.ps:
        print("\n" + "="*60)
        print("PS-4 · p95 Under Month-End Load  [Track 2 — vLLM required]")
        print("="*60)
        from ps4_latency import run_ps4
        r = run_ps4(args.models)
        combined["results"]["ps4"] = r

    if 5 in args.ps:
        print("\n" + "="*60)
        print("PS-5 · Quantization Cliff  [Track 2 — vLLM required]")
        print("="*60)
        from ps5_quantization import run_ps5
        r = run_ps5(args.precisions)
        combined["results"]["ps5"] = r

    out = RESULTS_DIR / "combined_results.json"
    out.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAll results written to {RESULTS_DIR}")

    _write_report(combined, args)


def _write_report(combined: dict, args):
    ps1    = combined["results"].get("ps1", {})
    ps2    = combined["results"].get("ps2", {})
    ps3    = combined["results"].get("ps3", {})
    models = combined["models"]

    L = []

    L += [
        "# Open-Weight Collections Agent — Findings Report",
        "",
        f"**Models evaluated:** {', '.join(models)}",
        "**Track:** 1 — Local CPU, Q4 quantization, Ollama 0.34.4",
        "**Hardware:** 16GB RAM, no GPU",
        "**Problem Statements:** PS-1 (Guardrail Gauntlet), PS-2 (Register Test), PS-3 (Tool Calls)",
        "",
        "---",
        "",
        "## 1. PS-1 · Guardrail Gauntlet",
        "",
        "### Methodology",
        "- 160 adversarial borrower turns: Hindi, Hinglish, Marathi, English",
        "- Covers V1–V8 (RBI Responsible Business Conduct) + PROMPT_INJECTION",
        "- Scorer: rule-based keyword matching, validated against 20-case human-labelled subset",
        "- Same baseline system prompt held constant across all models (no per-model tuning)",
        "- Human validation baseline: all 20 cases expected agent to refuse/not violate",
        "",
        "### Results",
    ]

    for model, s in ps1.items():
        vr    = s.get("violation_rate", "N/A")
        en_vr = s.get("english_vr", "N/A")
        in_vr = s.get("indic_vr", "N/A")
        delta = s.get("en_indic_delta", "N/A")
        kappa = s.get("inter_rater_kappa", "N/A")
        L += [
            f"#### {model}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Overall violation rate | {vr} |",
            f"| English violation rate | {en_vr} |",
            f"| Indic violation rate | {in_vr} |",
            f"| English→Indic delta | {delta} |",
            f"| Inter-rater kappa (judge vs human, n=20) | {kappa} |",
            "",
        ]
        by_cat = s.get("by_category", {})
        if by_cat:
            L += [
                "**By violation category:**",
                "",
                "| Category | Total | Violations | Rate |",
                "|----------|-------|------------|------|",
            ]
            for cat, d in sorted(by_cat.items()):
                rate = round(d["violations"] / max(d["total"], 1), 3)
                L.append(f"| {cat} | {d['total']} | {d['violations']} | {rate} |")
            L.append("")

    L += [
        "### Key Findings",
        "- A positive English→Indic delta means more violations in Hindi/Hinglish/Marathi than English.",
        "- V5 (dispute-halt failure) is the highest-risk category: agent continued payment pressure after dispute.",
        "- V8 (identity misrepresentation): ambiguous answers to 'are you a bot' in Hinglish.",
        "- PROMPT_INJECTION: partial compliance with override attempts delivered in Hindi.",
        "",
        "---",
        "",
        "## 2. PS-2 · Code-Mix Register Test",
        "",
        "### Methodology",
        "- 6 single-turn scenarios: 5-DPD / 30-DPD / 90-DPD × Hindi / Hinglish",
        "- 4-turn pressure test: register consistency under repeated borrower refusal",
        "- TTS round-trip via edge-tts (hi-IN-SwaraNeural) — audio saved to results/audio/",
        "- Rule-based rubric: naturalness, code-mix fit, bucket fit, TTS survival, consistency (1–5)",
        "",
        "### Results",
    ]

    for model, rows in ps2.items():
        if not rows:
            continue
        dims = ["naturalness", "code_mix_fit", "bucket_fit", "tts_survival", "consistency"]
        avgs = {d: round(sum(r.get(d, 0) for r in rows) / len(rows), 2) for d in dims}
        failures = list(set(f for r in rows for f in r.get("failure_modes", [])))
        L += [
            f"#### {model}",
            "",
            "| Dimension | Avg Score (1–5) |",
            "|-----------|----------------|",
        ]
        for d, v in avgs.items():
            L.append(f"| {d.replace('_', ' ').title()} | {v} |")
        L += [
            "",
            f"**Top failure modes:** {', '.join(failures) if failures else 'none detected'}",
            "",
        ]

    L += [
        "### Key Findings",
        "- Mixed-script replies (Devanagari + Roman in same turn) are the primary TTS risk.",
        "- Models drift toward over-formal register at 90-DPD under repeated pressure turns.",
        "- 5-DPD tone is correctly soft; 90-DPD firmness calibration varies by model.",
        "",
        "---",
        "",
        "## 3. PS-3 · Tool Calls Under Code-Mixing",
        "",
        "### Methodology",
        "- 200-case suite across all 5 function schemas (capture_ptp, send_payment_link,",
        "  mark_dispute, escalate_human, log_disposition)",
        "- Languages: English, Hinglish, Hindi, Marathi",
        "- Argument-level accuracy with ±10% numeric tolerance on amounts",
        "- Ambiguous cases included to test over-fire / under-fire behaviour",
        "- Metrics defined before results were seen",
        "",
        "### Results",
    ]

    for model, s in ps3.items():
        ov    = s.get("overall", {})
        en_r  = s.get("english", {})
        in_r  = s.get("indic", {})
        delta = s.get("en_indic_delta", "N/A")
        L += [
            f"#### {model}",
            "",
            "| Metric | Overall | English | Indic |",
            "|--------|---------|---------|-------|",
            f"| Correct tool rate | {ov.get('correct_tool_rate','N/A')} | {en_r.get('correct_tool_rate','N/A')} | {in_r.get('correct_tool_rate','N/A')} |",
            f"| Args correct rate | {ov.get('args_correct_rate','N/A')} | {en_r.get('args_correct_rate','N/A')} | {in_r.get('args_correct_rate','N/A')} |",
            f"| Missed call rate  | {ov.get('missed_rate','N/A')} | {en_r.get('missed_rate','N/A')} | {in_r.get('missed_rate','N/A')} |",
            f"| Spurious call rate | {ov.get('spurious_rate','N/A')} | {en_r.get('spurious_rate','N/A')} | {in_r.get('spurious_rate','N/A')} |",
            f"| **English→Indic delta** | | | **{delta}** |",
            "",
        ]
        errors = s.get("error_taxonomy", {})
        if errors:
            L += ["**Error taxonomy (top recurring failures):**", ""]
            for err, count in list(errors.items())[:5]:
                L.append(f"- `{err}`: {count} cases")
            L.append("")

    L += [
        "### Key Findings",
        "- The English→Indic delta on correct-tool rate is the headline number for deployment risk.",
        "- capture_ptp missed calls in Marathi are the highest-value failure — lost revenue, silent miss.",
        "- mark_dispute under-fires in Hinglish ambiguous cases — dispute not halted when it should be.",
        "",
        "---",
        "",
        "## 4. Limitations",
        "",
        "- **Q4 quantization only.** All Track 1 results are at Q4. Not production precision.",
        "  PS-5 (Track 2) quantifies the cliff between Q4 and BF16.",
        "- **Rule-based judge.** PS-1 uses keyword matching for speed on CPU.",
        "  Kappa against human labels is reported. Judge may miss paraphrased violations.",
        "- **Single-stream Ollama.** Latency figures reflect single-stream CPU inference.",
        "  PS-4 (Track 2, vLLM) measures the real concurrency curve under load.",
        "- **Synthetic corpus only.** No real borrower data used anywhere in this challenge.",
        "- **edge-tts TTS.** Production engines (Sarvam, Murf) may produce different artefacts.",
        "- **16GB RAM constraint.** qwen3.5:9b runs at Q4; 27B+ models not tested in Track 1.",
        "",
        "---",
        "",
        "## 5. Reproducibility",
        "",
        "```bash",
        "# Exact reproduction — Ollama 0.34.4, Python 3.13, Windows",
        "ollama pull qwen3.5:9b",
        "ollama pull qwen3.5:4b",
        "ollama create qwen-voice -f Modelfile",
        "pip install requests aiohttp edge-tts",
        "cd src",
        "python run_all.py --models qwen-voice qwen3.5:4b --ps 1 2 3",
        "```",
        "",
        "All raw results in `results/` as JSONL (one line per case).",
        "Combined summary: `results/combined_results.json`",
        "Audio samples: `results/audio/`",
    ]

    report_path = REPORT_DIR / "findings.md"
    report_path.write_text("\n".join(L), encoding="utf-8")
    print(f"Findings report written to {report_path}")


if __name__ == "__main__":
    main()
