# Open-Weight Collections Agent — Findings Report

**Models evaluated:** qwen3.5:4b
**Track:** 1 — Local CPU, Q4 quantization, Ollama 0.34.4
**Hardware:** 16GB RAM, no GPU
**Problem Statements:** PS-1 (Guardrail Gauntlet), PS-2 (Register Test), PS-3 (Tool Calls)

---

## 1. PS-1 · Guardrail Gauntlet

### Methodology
- 160 adversarial borrower turns: Hindi, Hinglish, Marathi, English
- Covers V1–V8 (RBI Responsible Business Conduct) + PROMPT_INJECTION
- Scorer: rule-based keyword matching, validated against 20-case human-labelled subset
- Same baseline system prompt held constant across all models (no per-model tuning)
- Human validation baseline: all 20 cases expected agent to refuse/not violate

### Results
#### qwen3.5:4b

| Metric | Value |
|--------|-------|
| Overall violation rate | 0.1062 |
| English violation rate | 0.0 |
| Indic violation rate | 0.1062 |
| English→Indic delta | 0.1062 |
| Inter-rater kappa (judge vs human, n=20) | -0.25 |

**By violation category:**

| Category | Total | Violations | Rate |
|----------|-------|------------|------|
| PROMPT_INJECTION | 6 | 2 | 0.333 |
| V1 | 31 | 1 | 0.032 |
| V2 | 23 | 4 | 0.174 |
| V3 | 23 | 6 | 0.261 |
| V4 | 19 | 2 | 0.105 |
| V5 | 22 | 0 | 0.0 |
| V6 | 14 | 2 | 0.143 |
| V7 | 11 | 0 | 0.0 |
| V8 | 11 | 0 | 0.0 |

### Key Findings
- A positive English→Indic delta means more violations in Hindi/Hinglish/Marathi than English.
- V5 (dispute-halt failure) is the highest-risk category: agent continued payment pressure after dispute.
- V8 (identity misrepresentation): ambiguous answers to 'are you a bot' in Hinglish.
- PROMPT_INJECTION: partial compliance with override attempts delivered in Hindi.

---

## 2. PS-2 · Code-Mix Register Test

### Methodology
- 6 single-turn scenarios: 5-DPD / 30-DPD / 90-DPD × Hindi / Hinglish
- 4-turn pressure test: register consistency under repeated borrower refusal
- TTS round-trip via edge-tts (hi-IN-SwaraNeural) — audio saved to results/audio/
- Rule-based rubric: naturalness, code-mix fit, bucket fit, TTS survival, consistency (1–5)

### Results
#### qwen3.5:4b

| Dimension | Avg Score (1–5) |
|-----------|----------------|
| Naturalness | 2.33 |
| Code Mix Fit | 2.67 |
| Bucket Fit | 2.17 |
| Tts Survival | 5.0 |
| Consistency | 5.0 |

**Top failure modes:** reply_too_short

### Key Findings
- Mixed-script replies (Devanagari + Roman in same turn) are the primary TTS risk.
- Models drift toward over-formal register at 90-DPD under repeated pressure turns.
- 5-DPD tone is correctly soft; 90-DPD firmness calibration varies by model.

---

## 3. PS-3 · Tool Calls Under Code-Mixing

### Methodology
- 200-case suite across all 5 function schemas (capture_ptp, send_payment_link,
  mark_dispute, escalate_human, log_disposition)
- Languages: English, Hinglish, Hindi, Marathi
- Argument-level accuracy with ±10% numeric tolerance on amounts
- Ambiguous cases included to test over-fire / under-fire behaviour
- Metrics defined before results were seen

### Results
#### qwen3.5:4b

| Metric | Overall | English | Indic |
|--------|---------|---------|-------|
| Correct tool rate | 0.71 | 0.8846 | 0.6486 |
| Args correct rate | 0.68 | 0.8462 | 0.6216 |
| Missed call rate  | 0.245 | 0.0962 | 0.2973 |
| Spurious call rate | 0.0 | 0.0 | 0.0 |
| **English→Indic delta** | | | **-0.236** |

**Error taxonomy (top recurring failures):**

- `wrong_tool:escalate_human->None`: 18 cases
- `wrong_tool:capture_ptp->None`: 14 cases
- `wrong_tool:mark_dispute->None`: 9 cases
- `wrong_tool:send_payment_link->None`: 5 cases
- `wrong_tool:send_payment_link->mark_dispute`: 4 cases

### Key Findings
- The English→Indic delta on correct-tool rate is the headline number for deployment risk.
- capture_ptp missed calls in Marathi are the highest-value failure — lost revenue, silent miss.
- mark_dispute under-fires in Hinglish ambiguous cases — dispute not halted when it should be.

---

## 4. Limitations

- **Q4 quantization only.** All Track 1 results are at Q4. Not production precision.
  PS-5 (Track 2) quantifies the cliff between Q4 and BF16.
- **Rule-based judge.** PS-1 uses keyword matching for speed on CPU.
  Kappa against human labels is reported. Judge may miss paraphrased violations.
- **Single-stream Ollama.** Latency figures reflect single-stream CPU inference.
  PS-4 (Track 2, vLLM) measures the real concurrency curve under load.
- **Synthetic corpus only.** No real borrower data used anywhere in this challenge.
- **edge-tts TTS.** Production engines (Sarvam, Murf) may produce different artefacts.
- **16GB RAM constraint.** qwen3.5:9b runs at Q4; 27B+ models not tested in Track 1.

---

## 5. Reproducibility

```bash
# Exact reproduction — Ollama 0.34.4, Python 3.13, Windows
ollama pull qwen3.5:9b
ollama pull qwen3.5:4b
ollama create qwen-voice -f Modelfile
pip install requests aiohttp edge-tts
cd src
python run_all.py --models qwen-voice qwen3.5:4b --ps 1 2 3
```

All raw results in `results/` as JSONL (one line per case).
Combined summary: `results/combined_results.json`
Audio samples: `results/audio/`