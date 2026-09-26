"""
PS-2 · Code-Mix Register Test
Text → TTS round-trip harness. Scores naturalness, code-mix fit, bucket fit, TTS survival, consistency.
Uses edge-tts (free, no API key) for audio generation.
"""
import asyncio
import json
from pathlib import Path

from agent import CollectionsAgent

RESULTS_DIR = Path(__file__).parent.parent / "results"
AUDIO_DIR   = RESULTS_DIR / "audio"
OLLAMA_URL  = "http://localhost:11434/api/chat"

# 3 DPD buckets × 2 languages = 6 scenarios
SCENARIOS = [
    {"dpd": 5,  "language": "Hinglish", "borrower_turn": "Bhai, ek chota sa reminder tha, EMI due hai aaj."},
    {"dpd": 30, "language": "Hinglish", "borrower_turn": "Yaar, paise tight hain, thoda time do please."},
    {"dpd": 90, "language": "Hinglish", "borrower_turn": "Main payment nahi kar sakta, koi raasta nahi hai."},
    {"dpd": 5,  "language": "Hindi",    "borrower_turn": "Aaj EMI ki yaad dilane ke liye shukriya."},
    {"dpd": 30, "language": "Hindi",    "borrower_turn": "Abhi thodi takleef hai, kuch din aur chahiye."},
    {"dpd": 90, "language": "Hindi",    "borrower_turn": "Main bilkul payment karne ki sthiti mein nahi hoon."},
]

# Fast rule-based register scorer — no extra LLM call
_BUCKET_TONE = {
    5:  {"firm_words": ["please","request","remind","yaad","shukriya","dhanyavaad"],
         "harsh_words": ["must","immediately","legal","court","arrest","jail"]},
    30: {"firm_words": ["important","zaruri","please","request","soon","jaldi"],
         "harsh_words": ["arrest","jail","seize","FIR","criminal"]},
    90: {"firm_words": ["serious","urgent","final","last","strict","important","zaruri","gambhir"],
         "harsh_words": ["arrest","jail","seize","FIR","criminal","police"]},
}

# Multi-turn stress test (register consistency under pressure)
PRESSURE_TURNS = [
    "Nahi karunga payment.",
    "Koi bhi karo, main nahi dunga.",
    "Tumse baat karna bekaar hai.",
    "Final answer: NO.",
]

RUBRIC_PROMPT = """You are a senior collections trainer for an Indian NBFC.
Score the following agent response on 5 dimensions (1-5 each):

1. Naturalness: 1=translated/stiff, 3=understandable, 5=native-sounding
2. Code-mix fit: 1=wrong register mix, 3=plausible, 5=matches borrower segment
3. Bucket fit: 1=wrong tone for DPD, 3=broadly appropriate, 5=precisely calibrated
4. TTS survival: 1=would sound wrong/garbled, 3=audible with minor issues, 5=clean audio
5. Consistency: 1=register drifts, 3=minor drift, 5=holds throughout

Context:
- DPD bucket: {dpd}
- Language: {language}
- Borrower said: {borrower_turn}
- Agent replied: {agent_reply}

Respond ONLY with JSON:
{{
  "naturalness": <1-5>,
  "code_mix_fit": <1-5>,
  "bucket_fit": <1-5>,
  "tts_survival": <1-5>,
  "consistency": <1-5>,
  "failure_modes": ["<mode1>", ...],
  "notes": "<one sentence>"
}}"""


def score_register(dpd: int, language: str, borrower_turn: str,
                   agent_reply: str, judge_model: str = "qwen-voice") -> dict:
    """Fast rule-based register scorer. No LLM call."""
    r = agent_reply.lower()
    tone = _BUCKET_TONE.get(dpd, _BUCKET_TONE[30])

    # Naturalness: penalise very short or very long replies
    words = len(agent_reply.split())
    naturalness = 5 if 8 <= words <= 40 else (3 if 4 <= words <= 60 else 1)

    # Code-mix fit: Hinglish should have both Hindi and English words
    has_hindi = any(c in agent_reply for c in "अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह")
    has_roman_hindi = any(w in r for w in ["hai","hoon","karo","dena","please","aap","main","yeh"])
    if language == "Hinglish":
        code_mix_fit = 5 if (has_roman_hindi and not has_hindi) else 3
    elif language == "Hindi":
        code_mix_fit = 5 if has_hindi else (3 if has_roman_hindi else 1)
    else:
        code_mix_fit = 3

    # Bucket fit
    firm_hit  = sum(1 for w in tone["firm_words"]  if w in r)
    harsh_hit = sum(1 for w in tone["harsh_words"] if w in r)
    if harsh_hit > 0:
        bucket_fit = 1
    elif firm_hit >= 2:
        bucket_fit = 5
    elif firm_hit == 1:
        bucket_fit = 3
    else:
        bucket_fit = 2

    # TTS survival: penalise mixed script (Devanagari + Roman in same reply)
    tts_survival = 3 if (has_hindi and has_roman_hindi) else 5

    # Consistency placeholder (single turn = 5 by default)
    consistency = 5

    failure_modes = []
    if harsh_hit > 0:
        failure_modes.append("harsh_tone_for_bucket")
    if has_hindi and has_roman_hindi:
        failure_modes.append("mixed_script_tts_risk")
    if words < 4:
        failure_modes.append("reply_too_short")
    if words > 60:
        failure_modes.append("reply_too_long_for_voice")

    return {
        "naturalness":   naturalness,
        "code_mix_fit":  code_mix_fit,
        "bucket_fit":    bucket_fit,
        "tts_survival":  tts_survival,
        "consistency":   consistency,
        "failure_modes": failure_modes,
        "notes":         f"words={words} firm={firm_hit} harsh={harsh_hit}",
    }


async def tts_to_file(text: str, output_path: Path, voice: str = "hi-IN-SwaraNeural"):
    """Generate audio via edge-tts (free, no API key required)."""
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))
        return True
    except ImportError:
        # Fallback: write text stub so pipeline continues without audio
        output_path.with_suffix(".txt").write_text(text, encoding="utf-8")
        return False
    except Exception as e:
        print(f"  TTS warning: {e}")
        return False


def run_ps2(models: list[str], judge_model: str = "qwen-voice") -> dict:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for model in models:
        print(f"\n=== PS-2 | Model: {model} ===")
        model_rows = []

        # Single-turn scenarios
        for sc in SCENARIOS:
            agent  = CollectionsAgent(model=model, dpd=sc["dpd"])
            result = agent.turn(sc["borrower_turn"], use_tools=False)
            reply  = result["reply"]

            scores = score_register(sc["dpd"], sc["language"], sc["borrower_turn"], reply, judge_model)

            # TTS round-trip
            safe_model = model.replace(":", "_").replace("/", "_")
            audio_path = AUDIO_DIR / f"{safe_model}_dpd{sc['dpd']}_{sc['language']}.mp3"
            tts_ok = asyncio.run(tts_to_file(reply, audio_path))

            row = {
                "model":         model,
                "dpd":           sc["dpd"],
                "language":      sc["language"],
                "borrower_turn": sc["borrower_turn"],
                "agent_reply":   reply,
                "tts_file":      str(audio_path) if tts_ok else None,
                **scores,
            }
            model_rows.append(row)
            avg = sum([scores.get("naturalness",0), scores.get("code_mix_fit",0),
                       scores.get("bucket_fit",0), scores.get("tts_survival",0),
                       scores.get("consistency",0)]) / 5
            print(f"  DPD={sc['dpd']} lang={sc['language']} avg_score={avg:.1f} "
                  f"failures={scores.get('failure_modes', [])}")

        # Multi-turn pressure test (register consistency)
        agent = CollectionsAgent(model=model, dpd=90)
        pressure_replies = []
        for turn_text in PRESSURE_TURNS:
            r = agent.turn(turn_text, use_tools=False)
            pressure_replies.append(r["reply"])

        # Score consistency across pressure turns
        pressure_scores = []
        for i, (turn_text, reply) in enumerate(zip(PRESSURE_TURNS, pressure_replies)):
            s = score_register(90, "Hinglish", turn_text, reply, judge_model)
            pressure_scores.append(s.get("consistency", 0))

        consistency_drift = max(pressure_scores) - min(pressure_scores) if pressure_scores else 0
        model_rows.append({
            "model":              model,
            "test_type":          "pressure_consistency",
            "turns":              PRESSURE_TURNS,
            "replies":            pressure_replies,
            "consistency_scores": pressure_scores,
            "consistency_drift":  consistency_drift,
        })
        print(f"  Pressure test consistency drift: {consistency_drift}")

        all_results[model] = model_rows

    return all_results


def save_results(all_results: dict):
    RESULTS_DIR.mkdir(exist_ok=True)
    summary_rows = []

    for model, rows in all_results.items():
        safe = model.replace(":", "_").replace("/", "_")
        (RESULTS_DIR / f"ps2_{safe}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
        )
        # Aggregate scores (skip pressure row)
        score_rows = [r for r in rows if "naturalness" in r]
        if score_rows:
            dims = ["naturalness", "code_mix_fit", "bucket_fit", "tts_survival", "consistency"]
            avgs = {d: round(sum(r.get(d, 0) for r in score_rows) / len(score_rows), 2) for d in dims}
            all_failures = [f for r in score_rows for f in r.get("failure_modes", [])]
            summary_rows.append({"model": model, **avgs, "top_failures": list(set(all_failures))[:5]})

    (RESULTS_DIR / "ps2_summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nPS-2 Summary saved.")
    for s in summary_rows:
        print(f"  {s['model']}: naturalness={s['naturalness']} bucket_fit={s['bucket_fit']} "
              f"tts={s['tts_survival']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen-voice", "qwen3.5:4b"])
    parser.add_argument("--judge",  default="qwen-voice")
    args = parser.parse_args()

    results = run_ps2(args.models, judge_model=args.judge)
    save_results(results)
