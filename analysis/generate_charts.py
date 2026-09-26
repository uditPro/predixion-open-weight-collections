"""
analysis/generate_charts.py
Run after overnight eval to produce all charts and summary tables.
Usage: python analysis/generate_charts.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
REPORT_DIR  = Path(__file__).parent.parent / "report"
ANALYSIS_DIR = Path(__file__).parent
CHARTS_DIR  = ANALYSIS_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def ps1_tables():
    summary = load_json(RESULTS_DIR / "ps1_summary.json")
    if not summary:
        print("PS-1 summary not found, skipping.")
        return

    print("\n" + "="*70)
    print("PS-1 · GUARDRAIL GAUNTLET — RESULTS TABLE")
    print("="*70)

    # Model comparison table
    print(f"\n{'Model':<30} {'ViolRate':>10} {'EN_VR':>8} {'IN_VR':>8} {'Delta':>8} {'Kappa':>8}")
    print("-"*70)
    for model, s in summary.items():
        print(f"{model:<30} {s.get('violation_rate',0):>10.4f} "
              f"{s.get('english_vr',0):>8.4f} {s.get('indic_vr',0):>8.4f} "
              f"{s.get('en_indic_delta',0):>8.4f} {str(s.get('inter_rater_kappa','N/A')):>8}")

    # By category
    print("\n--- Violation Rate by Category ---")
    cats = set()
    for s in summary.values():
        cats.update(s.get("by_category", {}).keys())
    cats = sorted(cats)

    header = f"{'Category':<20}" + "".join(f"{m[:12]:>14}" for m in summary.keys())
    print(header)
    print("-" * len(header))
    for cat in cats:
        row = f"{cat:<20}"
        for s in summary.values():
            d = s.get("by_category", {}).get(cat, {})
            rate = d.get("violations", 0) / max(d.get("total", 1), 1)
            row += f"{rate:>14.3f}"
        print(row)

    # By language
    print("\n--- Violation Rate by Language ---")
    langs = set()
    for s in summary.values():
        langs.update(s.get("by_language", {}).keys())
    langs = sorted(langs)

    header = f"{'Language':<15}" + "".join(f"{m[:12]:>14}" for m in summary.keys())
    print(header)
    print("-" * len(header))
    for lang in langs:
        row = f"{lang:<15}"
        for s in summary.values():
            d = s.get("by_language", {}).get(lang, {})
            rate = d.get("violations", 0) / max(d.get("total", 1), 1)
            row += f"{rate:>14.3f}"
        print(row)


def ps3_tables():
    summary = load_json(RESULTS_DIR / "ps3_summary.json")
    if not summary:
        print("PS-3 summary not found, skipping.")
        return

    print("\n" + "="*70)
    print("PS-3 · TOOL CALLS — RESULTS TABLE")
    print("="*70)

    print(f"\n{'Model':<30} {'ToolRate':>10} {'ArgsRate':>10} {'Missed':>8} {'Spurious':>10} {'Delta':>8}")
    print("-"*70)
    for model, s in summary.items():
        ov = s.get("overall", {})
        print(f"{model:<30} {ov.get('correct_tool_rate',0):>10.4f} "
              f"{ov.get('args_correct_rate',0):>10.4f} "
              f"{ov.get('missed_rate',0):>8.4f} "
              f"{ov.get('spurious_rate',0):>10.4f} "
              f"{s.get('en_indic_delta',0):>8.4f}")

    print("\n--- English vs Indic Breakdown ---")
    print(f"\n{'Model':<30} {'EN_Tool':>10} {'IN_Tool':>10} {'EN_Args':>10} {'IN_Args':>10}")
    print("-"*70)
    for model, s in summary.items():
        en = s.get("english", {})
        ind = s.get("indic", {})
        print(f"{model:<30} {en.get('correct_tool_rate',0):>10.4f} "
              f"{ind.get('correct_tool_rate',0):>10.4f} "
              f"{en.get('args_correct_rate',0):>10.4f} "
              f"{ind.get('args_correct_rate',0):>10.4f}")

    print("\n--- Error Taxonomy ---")
    for model, s in summary.items():
        print(f"\n{model}:")
        for err, count in list(s.get("error_taxonomy", {}).items())[:5]:
            print(f"  {err}: {count}")


def ps2_tables():
    summary_path = RESULTS_DIR / "ps2_summary.json"
    if not summary_path.exists():
        print("PS-2 summary not found, skipping.")
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print("\n" + "="*70)
    print("PS-2 · REGISTER TEST — RESULTS TABLE")
    print("="*70)

    dims = ["naturalness", "code_mix_fit", "bucket_fit", "tts_survival", "consistency"]
    header = f"{'Model':<30}" + "".join(f"{d[:10]:>12}" for d in dims)
    print(header)
    print("-"*len(header))
    for s in summary:
        row = f"{s.get('model',''):<30}"
        for d in dims:
            row += f"{s.get(d, 0):>12.2f}"
        print(row)
        print(f"  Failures: {s.get('top_failures', [])}")


def failure_taxonomy():
    """Print top failure patterns across all PS."""
    print("\n" + "="*70)
    print("FAILURE TAXONOMY — TOP PATTERNS")
    print("="*70)

    for model_file in RESULTS_DIR.glob("ps1_*.jsonl"):
        rows = load_jsonl(model_file)
        violations = [r for r in rows if r.get("violated")]
        if not violations:
            continue
        model = model_file.stem.replace("ps1_", "")
        print(f"\n{model} — {len(violations)} violations:")
        cats: dict[str, int] = {}
        for r in violations:
            c = r.get("violation_code") or r.get("violation_target", "unknown")
            cats[c] = cats.get(c, 0) + 1
        for c, n in sorted(cats.items(), key=lambda x: -x[1]):
            print(f"  {c}: {n}")

    for model_file in RESULTS_DIR.glob("ps3_*.jsonl"):
        rows = load_jsonl(model_file)
        fails = [r for r in rows if not r.get("correct_tool") or not r.get("args_correct")]
        if not fails:
            continue
        model = model_file.stem.replace("ps3_", "")
        print(f"\n{model} PS-3 failures: {len(fails)}/{len(rows)}")
        errs: dict[str, int] = {}
        for r in fails:
            key = f"{r.get('expected_tool')}→{r.get('actual_tool')}"
            errs[key] = errs.get(key, 0) + 1
        for k, n in sorted(errs.items(), key=lambda x: -x[1])[:5]:
            print(f"  {k.replace(chr(8594), '->')}: {n}")


def write_csv_summary():
    """Write summary.csv for easy spreadsheet review."""
    rows = []
    ps1 = load_json(RESULTS_DIR / "ps1_summary.json")
    ps3 = load_json(RESULTS_DIR / "ps3_summary.json")

    for model in set(list(ps1.keys()) + list(ps3.keys())):
        s1 = ps1.get(model, {})
        s3 = ps3.get(model, {})
        rows.append({
            "model":              model,
            "ps1_violation_rate": s1.get("violation_rate", ""),
            "ps1_en_vr":          s1.get("english_vr", ""),
            "ps1_indic_vr":       s1.get("indic_vr", ""),
            "ps1_en_indic_delta": s1.get("en_indic_delta", ""),
            "ps1_kappa":          s1.get("inter_rater_kappa", ""),
            "ps3_tool_rate":      s3.get("overall", {}).get("correct_tool_rate", ""),
            "ps3_args_rate":      s3.get("overall", {}).get("args_correct_rate", ""),
            "ps3_en_indic_delta": s3.get("en_indic_delta", ""),
        })

    if rows:
        import csv
        out = RESULTS_DIR / "summary.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\nSummary CSV written to {out}")


def chart_ps1_violations():
    """Bar chart: violation rate by category."""
    summary = load_json(RESULTS_DIR / "ps1_summary.json")
    if not summary:
        return
    model = list(summary.keys())[0]
    s = summary[model]
    cats = sorted(s.get("by_category", {}).keys())
    rates = []
    for cat in cats:
        d = s["by_category"][cat]
        rates.append(d.get("violations", 0) / max(d.get("total", 1), 1))

    plt.figure(figsize=(10, 5))
    bars = plt.bar(cats, rates, color="steelblue", edgecolor="black")
    plt.title(f"PS-1: Violation Rate by Category — {model}")
    plt.ylabel("Violation Rate")
    plt.ylim(0, max(rates)*1.3 if rates else 1)
    for bar, rate in zip(bars, rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{rate:.1%}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out = CHARTS_DIR / "ps1_violations_by_category.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Chart saved: {out}")


def chart_ps1_languages():
    """Bar chart: violation rate by language."""
    summary = load_json(RESULTS_DIR / "ps1_summary.json")
    if not summary:
        return
    model = list(summary.keys())[0]
    s = summary[model]
    langs = sorted(s.get("by_language", {}).keys())
    rates = []
    for lang in langs:
        d = s["by_language"][lang]
        rates.append(d.get("violations", 0) / max(d.get("total", 1), 1))

    plt.figure(figsize=(8, 5))
    bars = plt.bar(langs, rates, color="coral", edgecolor="black")
    plt.title(f"PS-1: Violation Rate by Language — {model}")
    plt.ylabel("Violation Rate")
    plt.ylim(0, max(rates)*1.3 if rates else 1)
    for bar, rate in zip(bars, rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{rate:.1%}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out = CHARTS_DIR / "ps1_violations_by_language.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Chart saved: {out}")


def chart_ps2_register():
    """Radar chart: PS-2 register dimensions."""
    summary_path = RESULTS_DIR / "ps2_summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary:
        return
    s = summary[0]
    model = s.get("model", "")
    dims = ["naturalness", "code_mix_fit", "bucket_fit", "tts_survival", "consistency"]
    values = [s.get(d, 0) for d in dims]

    angles = np.linspace(0, 2*np.pi, len(dims), endpoint=False).tolist()
    values_plot = values + values[:1]
    angles_plot = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles_plot, values_plot, "o-", linewidth=2, color="steelblue")
    ax.fill(angles_plot, values_plot, alpha=0.25, color="steelblue")
    ax.set_xticks(angles)
    ax.set_xticklabels([d.replace("_", " ").title() for d in dims])
    ax.set_ylim(0, 5)
    ax.set_title(f"PS-2: Register Scores — {model}", pad=20)
    ax.grid(True)
    plt.tight_layout()
    out = CHARTS_DIR / "ps2_register_radar.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Chart saved: {out}")


def chart_ps3_tool_accuracy():
    """Grouped bar chart: PS-3 tool accuracy English vs Indic."""
    summary = load_json(RESULTS_DIR / "ps3_summary.json")
    if not summary:
        return
    model = list(summary.keys())[0]
    s = summary[model]
    en = s.get("english", {})
    ind = s.get("indic", {})
    categories = ["Correct Tool", "Args Correct", "Missed", "Spurious"]
    en_vals = [en.get("correct_tool_rate", 0), en.get("args_correct_rate", 0),
               en.get("missed_rate", 0), en.get("spurious_rate", 0)]
    ind_vals = [ind.get("correct_tool_rate", 0), ind.get("args_correct_rate", 0),
                ind.get("missed_rate", 0), ind.get("spurious_rate", 0)]

    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(10, 5))
    bars1 = plt.bar(x - width/2, en_vals, width, label="English", color="steelblue", edgecolor="black")
    bars2 = plt.bar(x + width/2, ind_vals, width, label="Indic (Hindi/Hinglish/Marathi)", color="coral", edgecolor="black")
    plt.title(f"PS-3: Tool Call Accuracy — {model}")
    plt.ylabel("Rate")
    plt.xticks(x, categories)
    plt.ylim(0, 1.1)
    plt.legend()
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                plt.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                         f"{h:.0%}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out = CHARTS_DIR / "ps3_tool_accuracy.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Chart saved: {out}")


def chart_ps3_error_taxonomy():
    """Horizontal bar chart: top error types."""
    summary = load_json(RESULTS_DIR / "ps3_summary.json")
    if not summary:
        return
    model = list(summary.keys())[0]
    s = summary[model]
    errors = s.get("error_taxonomy", {})
    if not errors:
        return
    top = list(errors.items())[:10]
    labels = [e[0].replace("wrong_tool:", "").replace("bad_args:", "") for e in top]
    counts = [e[1] for e in top]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(labels[::-1], counts[::-1], color="mediumseagreen", edgecolor="black")
    plt.title(f"PS-3: Top Error Types — {model}")
    plt.xlabel("Count")
    for bar, count in zip(bars, counts[::-1]):
        plt.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 str(count), va="center", fontsize=9)
    plt.tight_layout()
    out = CHARTS_DIR / "ps3_error_taxonomy.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Chart saved: {out}")


def generate_all_charts():
    """Generate all visual charts."""
    print("\n" + "="*60)
    print("GENERATING VISUAL CHARTS")
    print("="*60)
    chart_ps1_violations()
    chart_ps1_languages()
    chart_ps2_register()
    chart_ps3_tool_accuracy()
    chart_ps3_error_taxonomy()
    print(f"\nAll charts saved to {CHARTS_DIR}")


if __name__ == "__main__":
    ps1_tables()
    ps2_tables()
    ps3_tables()
    failure_taxonomy()
    write_csv_summary()
    generate_all_charts()
    print("\nDone. Check results/summary.csv, analysis/charts/, and report/findings.md")
