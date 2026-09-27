# Predixion Open-Weight Collections Agent — Track 1

End-to-end evaluation of open-weight models for RBI-compliant Hindi/Hinglish collections voice calls.

## Hardware
- Track 1: 16GB RAM laptop, no GPU, Ollama
- Hosted API baseline: Sarvam-M (India-hosted, declared per challenge rules)

## Setup

```bash
# 1. Install Ollama from https://ollama.com, then:
ollama pull qwen3.5:9b
ollama pull qwen3.5:4b

# 2. Create voice model (thinking mode OFF — required by the challenge spec)
ollama create qwen-voice -f Modelfile

# 3. Python deps
pip install -r requirements.txt

# 4. Hosted baseline credentials (Sarvam-M)
#    Copy .env.example to .env and fill in SARVAM_API_KEY.
```

## Methodology notes (read before re-running)

- **Thinking mode is disabled at the API layer** (`think: false` as a top-level
  Ollama request parameter — placing it inside `options` is silently ignored and
  leaves thinking ON, inflating token counts and invalidating latency figures).
- **Hybrid judge** (`hybrid-v2.0`): a refusal-context-aware rule layer — forbidden
  content counts only *outside* refusal constructions (quoting a rule while
  refusing is compliance) — with an LLM-as-judge fallback for ambiguous cases.
- **V5 dispute-halt** is scored from tool behaviour: continued pressure after a
  clear dispute AND no `mark_dispute` call.
- Judge verdicts are **version-stamped**; rows from an older judge version are
  auto-invalidated at run start (archived as `*.invalid_*.jsonl`).
- Judge validated against reply-level human labels in `data/human_labels.json`;
  Cohen's kappa is reported in `results/ps1_summary.json`.

## Run

```bash
cd src
# Full Track 1: both open models, PS-1 + PS-2 + PS-3
python run_all.py --models qwen-voice qwen3.5:4b --ps 1 2 3

# Hosted baseline (Sarvam-M via .env)
python ps1_guardrail.py --models sarvam-m

# Individual PS
python ps1_guardrail.py --models qwen-voice qwen3.5:4b
python ps2_register.py  --models qwen-voice qwen3.5:4b
python ps3_tool_calls.py --models qwen-voice qwen3.5:4b

# Re-score existing replies without new model calls
python ps1_guardrail.py --models qwen3.5:4b --rejudge
```

## Results

| File | Contents |
|------|----------|
| `results/ps1_<model>.jsonl` | Per-case guardrail results (judge-version stamped) |
| `results/ps1_summary.json` | Violation rates by model, category, language + kappa |
| `results/ps2_<model>.jsonl` | Register scores per scenario |
| `results/ps3_<model>.jsonl` | Tool-call accuracy per case |
| `results/combined_results.json` | All summaries in one file |
| `results/audio/` | TTS audio samples (PS-2) |
| `results/archive_thinking_on_20260927/` | Pre-fix outputs, retained for provenance (thinking-mode bug) |

Findings report (auto-generated from data, no asserted findings): `report/findings.md`

## Models

| Model | Quantization | Role |
|-------|-------------|------|
| qwen3.5:9b (qwen-voice) | Q4_K_M | Primary Track 1 candidate |
| qwen3.5:4b | Q4 | Smallest viable |
| sarvam-m | hosted API | India-hosted baseline (challenge-declared) |

## Limitations

- Q4 results are not production conclusions (PS-5 on Track 2 quantifies the cliff).
- Hybrid judge: rule layer is deterministic but pattern-based; LLM fallback
  inherits judge-model bias — quantified via kappa, not eliminated.
- Single-stream Ollama latency; PS-4 (Track 2, vLLM) measures concurrency curves.
- Synthetic data only — no real borrower data anywhere in this challenge.
- edge-tts TTS; production engines may produce different artefacts.

## License

Apache 2.0
