"""
PS-5 · The Quantization Cliff
Re-runs PS-1 and PS-3 suites at Q4, Q8, FP8, BF16.
Locates the precision cliff for guardrail adherence and structured-output validity.
Requires vLLM (Track 2) with different model checkpoints loaded per precision.
"""
import json
from pathlib import Path

from ps1_guardrail import run_ps1, compute_summary as ps1_summary
from ps3_tool_calls import run_ps3, compute_ps3_summary

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Map precision label -> model identifier (update with actual vLLM model paths)
PRECISION_MODELS = {
    "Q4":   "qwen-voice",                    # Ollama Q4 (Track 1)
    "Q8":   "qwen3.5:9b-instruct-q8_0",      # Ollama Q8
    "FP8":  "Qwen/Qwen3.5-9B-FP8",           # vLLM FP8
    "BF16": "Qwen/Qwen3.5-9B",               # vLLM BF16
}


def run_ps5(precisions: list[str] | None = None, backend_map: dict | None = None):
    """
    precisions: subset of ["Q4","Q8","FP8","BF16"] to run (default all)
    backend_map: {"Q4":"ollama","Q8":"ollama","FP8":"vllm","BF16":"vllm"}
    """
    if precisions is None:
        precisions = list(PRECISION_MODELS.keys())
    if backend_map is None:
        backend_map = {"Q4": "ollama", "Q8": "ollama", "FP8": "vllm", "BF16": "vllm"}

    cliff_results: dict[str, dict] = {}

    for prec in precisions:
        model   = PRECISION_MODELS[prec]
        backend = backend_map.get(prec, "ollama")
        print(f"\n=== PS-5 | Precision: {prec} | Model: {model} | Backend: {backend} ===")

        # PS-1 guardrail
        ps1_raw  = run_ps1([model], judge_model="qwen-voice")
        ps1_sum  = ps1_summary(ps1_raw)

        # PS-3 tool calls
        ps3_raw  = run_ps3([model])
        ps3_sum  = compute_ps3_summary(ps3_raw)

        cliff_results[prec] = {
            "model":   model,
            "backend": backend,
            "ps1": {
                "violation_rate": ps1_sum[model]["violation_rate"],
                "en_indic_delta": ps1_sum[model]["en_indic_delta"],
            },
            "ps3": {
                "correct_tool_rate": ps3_sum[model]["overall"]["correct_tool_rate"],
                "args_correct_rate": ps3_sum[model]["overall"]["args_correct_rate"],
                "en_indic_delta":    ps3_sum[model]["en_indic_delta"],
            },
        }

    _save_cliff(cliff_results)
    _print_cliff_table(cliff_results)
    return cliff_results


def _save_cliff(cliff_results: dict):
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "ps5_quantization_cliff.json").write_text(
        json.dumps(cliff_results, indent=2), encoding="utf-8"
    )
    print(f"\nSaved ps5_quantization_cliff.json")


def _print_cliff_table(cliff_results: dict):
    print("\n--- Quantization Cliff Table ---")
    print(f"{'Prec':<6} {'ViolRate':>10} {'ToolRate':>10} {'ArgsRate':>10} {'EnIndicDelta(PS3)':>18}")
    for prec, d in cliff_results.items():
        print(f"{prec:<6} "
              f"{d['ps1']['violation_rate']:>10.4f} "
              f"{d['ps3']['correct_tool_rate']:>10.4f} "
              f"{d['ps3']['args_correct_rate']:>10.4f} "
              f"{d['ps3']['en_indic_delta']:>18.4f}")

    # Locate cliff: largest drop in tool_rate between adjacent precisions
    prec_order = [p for p in ["Q4", "Q8", "FP8", "BF16"] if p in cliff_results]
    max_drop   = 0.0
    cliff_at   = None
    for i in range(1, len(prec_order)):
        lo = cliff_results[prec_order[i-1]]["ps3"]["correct_tool_rate"]
        hi = cliff_results[prec_order[i]]["ps3"]["correct_tool_rate"]
        drop = hi - lo
        if drop > max_drop:
            max_drop = drop
            cliff_at = f"{prec_order[i-1]}->{prec_order[i]}"

    if cliff_at:
        print(f"\nLargest tool-rate improvement: {cliff_at} (+{max_drop:.4f})")
        print("Cliff location: precision below this level degrades structured output most.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--precisions", nargs="+", default=["Q4", "Q8"])
    args = parser.parse_args()
    run_ps5(args.precisions)
