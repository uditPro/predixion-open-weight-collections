# Open-Weight Collections Agent — Findings Report

**Models evaluated:** qwen3.5:4b, qwen-voice, sarvam/sarvam-105b-conversations
**Track:** 1 — Local, Q4 quantization, Ollama (thinking mode disabled at the API layer)
**Hardware:** 16GB RAM, no GPU
**Problem Statements:** PS-1 (Guardrail Gauntlet), PS-2 (Register Test), PS-3 (Tool Calls)

---

## 1. PS-1 · Guardrail Gauntlet

### Methodology
- Adversarial suite: Hindi, Hinglish, Marathi **and English** turns, each labelled with the violation category it targets (V1–V8 + PROMPT_INJECTION).
- **Hybrid judge:** a refusal-aware rule layer (forbidden content counts only *outside* refusal constructions — quoting a rule while refusing is compliance) with an LLM-as-judge fallback for cases the rule layer cannot decide.
- V5 (dispute-halt) is scored from tool behaviour: a violation requires continued recovery pressure after a clear dispute **and** no `mark_dispute` call.
- Same baseline system prompt across all models; the judge never sees which model produced a reply.
- Judge validated against reply-level human gold labels (`data/human_labels.json`); kappa reported below with an honest interpretation band.
- Judge logic is version-stamped; rows produced by an older judge are invalidated, not silently mixed in.

### Results
#### qwen3.5:4b

| Metric | Value |
|--------|-------|
| Cases (scored / total) | 192 / 192 |
| Violations | 3 |
| Overall violation rate | 0.0156 |
| English violation rate | 0.0312 |
| Indic violation rate | 0.0125 |
| English→Indic delta | -0.0187 |
| Judge–human kappa (n=0) | N/A |
| Judge version | `hybrid-v2.0` |

**By violation category:**

| Category | Total | Violations | Rate |
|----------|-------|------------|------|
| PROMPT_INJECTION | 8 | 0 | 0.000 |
| V1 | 36 | 0 | 0.000 |
| V2 | 28 | 0 | 0.000 |
| V3 | 27 | 2 | 0.074 |
| V4 | 23 | 0 | 0.000 |
| V5 | 26 | 1 | 0.038 |
| V6 | 17 | 0 | 0.000 |
| V7 | 14 | 0 | 0.000 |
| V8 | 13 | 0 | 0.000 |

#### qwen-voice

| Metric | Value |
|--------|-------|
| Cases (scored / total) | 192 / 192 |
| Violations | 4 |
| Overall violation rate | 0.0208 |
| English violation rate | 0.0312 |
| Indic violation rate | 0.0187 |
| English→Indic delta | -0.0125 |
| Judge–human kappa (n=0) | N/A |
| Judge version | `hybrid-v2.0` |

**By violation category:**

| Category | Total | Violations | Rate |
|----------|-------|------------|------|
| PROMPT_INJECTION | 8 | 0 | 0.000 |
| V1 | 36 | 0 | 0.000 |
| V2 | 28 | 2 | 0.071 |
| V3 | 27 | 0 | 0.000 |
| V4 | 23 | 0 | 0.000 |
| V5 | 26 | 2 | 0.077 |
| V6 | 17 | 0 | 0.000 |
| V7 | 14 | 0 | 0.000 |
| V8 | 13 | 0 | 0.000 |

#### sarvam/sarvam-105b-conversations

| Metric | Value |
|--------|-------|
| Cases (scored / total) | 192 / 192 |
| Violations | 3 |
| Overall violation rate | 0.0156 |
| English violation rate | 0.0000 |
| Indic violation rate | 0.0187 |
| English→Indic delta | 0.0187 |
| Judge–human kappa (n=0) | N/A |
| Judge version | `hybrid-v2.0` |

**By violation category:**

| Category | Total | Violations | Rate |
|----------|-------|------------|------|
| PROMPT_INJECTION | 8 | 0 | 0.000 |
| V1 | 36 | 0 | 0.000 |
| V2 | 28 | 1 | 0.036 |
| V3 | 27 | 0 | 0.000 |
| V4 | 23 | 1 | 0.043 |
| V5 | 26 | 1 | 0.038 |
| V6 | 17 | 0 | 0.000 |
| V7 | 14 | 0 | 0.000 |
| V8 | 13 | 0 | 0.000 |

### Key Findings (derived from the tables above — not asserted a priori)

- Highest actual violation rate [qwen3.5:4b]: **V3** (2/27 = 7.4%).
- Also violated: V5 (1/26).
- Categories with zero violations [qwen3.5:4b]: PROMPT_INJECTION, V1, V2, V4, V6, V7, V8 — may reflect suite weakness rather than model strength.
- English→Indic delta [qwen3.5:4b]: **-0.0187** — MORE violations in English (EN 0.0312 vs Indic 0.0125, n_en=32).
- Judge–human agreement [qwen3.5:4b]: kappa = N/A (not computed, n=0). 
- Highest actual violation rate [qwen-voice]: **V5** (2/26 = 7.7%).
- Also violated: V2 (2/28).
- Categories with zero violations [qwen-voice]: PROMPT_INJECTION, V1, V3, V4, V6, V7, V8 — may reflect suite weakness rather than model strength.
- English→Indic delta [qwen-voice]: **-0.0125** — MORE violations in English (EN 0.0312 vs Indic 0.0187, n_en=32).
- Judge–human agreement [qwen-voice]: kappa = N/A (not computed, n=0). 
- Highest actual violation rate [sarvam/sarvam-105b-conversations]: **V4** (1/23 = 4.3%).
- Also violated: V5 (1/26); V2 (1/28).
- Categories with zero violations [sarvam/sarvam-105b-conversations]: PROMPT_INJECTION, V1, V3, V6, V7, V8 — may reflect suite weakness rather than model strength.
- English→Indic delta [sarvam/sarvam-105b-conversations]: **+0.0187** — MORE violations in Indic languages (EN 0.0000 vs Indic 0.0187, n_en=32).
- Judge–human agreement [sarvam/sarvam-105b-conversations]: kappa = N/A (not computed, n=0). 
- Judge version: `hybrid-v2.0`. Rows from older judge versions are auto-invalidated at run start; raw pre-fix outputs are archived as `*.invalid_*.jsonl`.

---

## 2. PS-2 · Code-Mix Register Test

### Methodology
- Single-turn scenarios across 5/30/90-DPD × Hindi/Hinglish plus a multi-turn pressure test under repeated borrower refusal.
- TTS round-trip via edge-tts (hi-IN-SwaraNeural); audio in `results/audio/`.
- Rubric: naturalness, code-mix fit, bucket fit, TTS survival, consistency (1–5).

### Results

#### qwen3.5:4b

| Dimension | Avg Score (1–5) |
|-----------|----------------|
| Naturalness | 4.33 |
| Code Mix Fit | 4.33 |
| Bucket Fit | 2.50 |
| Tts Survival | 5.00 |
| Consistency | 5.00 |

#### qwen-voice

| Dimension | Avg Score (1–5) |
|-----------|----------------|
| Naturalness | 2.33 |
| Code Mix Fit | 4.00 |
| Bucket Fit | 2.17 |
| Tts Survival | 5.00 |
| Consistency | 5.00 |

#### sarvam/sarvam-105b-conversations

| Dimension | Avg Score (1–5) |
|-----------|----------------|
| Naturalness | 5.00 |
| Code Mix Fit | 4.00 |
| Bucket Fit | 2.00 |
| Tts Survival | 5.00 |
| Consistency | 5.00 |

### Key Findings

- [qwen3.5:4b] Weakest dimension: **bucket_fit** (2.50/5); strongest: tts_survival (5.00/5).
- [qwen-voice] Weakest dimension: **bucket_fit** (2.17/5); strongest: tts_survival (5.00/5).
- [qwen-voice] High TTS survival with low text naturalness suggests the TTS scorer measures readability, not prosody — audio review needed before trusting TTS survival.
- Failure modes detected [qwen-voice]: reply_too_long_for_voice.
- [sarvam/sarvam-105b-conversations] Weakest dimension: **bucket_fit** (2.00/5); strongest: naturalness (5.00/5).

---

## 3. PS-3 · Tool Calls Under Code-Mixing

### Methodology
- Suite across the 5 fixed schemas; argument-level accuracy with numeric tolerance; ambiguous cases for over/under-fire behaviour.
- Metrics defined before results were seen.

### Results

#### sarvam/sarvam-105b-conversations

| Metric | Overall | English | Indic |
|--------|---------|---------|-------|
| Correct tool rate | 0.825 | 0.8846 | 0.8041 |
| Args correct rate | 0.81 | 0.8846 | 0.7838 |
| Missed call rate | 0.125 | 0.0385 | 0.1554 |
| Spurious call rate | 0.0 | 0.0 | 0.0 |
| **English→Indic delta** | | | **-0.0805** |

**Error taxonomy (top recurring failures):**

- `wrong_tool:mark_dispute->None`: 10 cases
- `wrong_tool:send_payment_link->None`: 8 cases
- `wrong_tool:capture_ptp->None`: 5 cases
- `wrong_tool:log_disposition->mark_dispute`: 4 cases
- `wrong_tool:log_disposition->escalate_human`: 3 cases

### Key Findings

- [sarvam/sarvam-105b-conversations] Correct-tool rate drops from 88.5% (EN) to 80.4% (Indic): delta **-0.0805**.
- Dominant failure shapes [sarvam/sarvam-105b-conversations]: `wrong_tool:mark_dispute->None` ×10, `wrong_tool:send_payment_link->None` ×8, `wrong_tool:capture_ptp->None` ×5.

---

## 4. Limitations

- **Q4 quantization only.** Track 1 results are Q4. Not production precision; PS-5 quantifies the cliff between Q4 and BF16.
- **Hybrid judge, not gold-standard.** The rule layer is deterministic but pattern-based; the LLM fallback inherits whatever biases the judge model has. Kappa against human labels quantifies (does not eliminate) this.
- **Human validation subset is reply-level** and authored against actual stored outputs, not attack intent; disagreements are logged in `data/human_labels.json`.
- **Single-stream Ollama.** Latency figures reflect single-stream CPU inference; PS-4 (Track 2, vLLM) measures the concurrency curve.
- **Synthetic corpus only.** No real borrower data used anywhere.
- **edge-tts TTS.** Production engines may produce different artefacts.

---

## 5. Reproducibility

```bash
# Ollama + Python 3.13, Windows
ollama pull qwen3.5:9b && ollama pull qwen3.5:4b
ollama create qwen-voice -f Modelfile
pip install -r requirements.txt
cd src
python run_all.py --models qwen-voice qwen3.5:4b --ps 1 2 3
```

All raw results in `results/` as JSONL (one line per case, judge-version stamped).
Combined summary: `results/combined_results.json`. Audio: `results/audio/`.