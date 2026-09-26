# Predixion Open-Weight Collections Agent

End-to-end evaluation of open-weight models for RBI-compliant Hindi/Hinglish collections voice calls.

## Hardware
- Track 1: 16GB RAM, no GPU required
- Track 2: Single GPU (AWS ap-south-1, g5.xlarge), vLLM

## Setup

```bash
# 1. Install Ollama from https://ollama.com, then:
ollama pull qwen3.5:9b
ollama pull qwen3.5:4b

# 2. Create voice model (thinking mode OFF — required)
ollama create qwen-voice -f Modelfile

# 3. Install Python deps
pip install -r requirements.txt
```

## Run Track 1 (PS-1, PS-2, PS-3)

```bash
cd src
# All three problem statements, both models
python run_all.py --models qwen-voice qwen3.5:4b --ps 1 2 3

# Individual PS
python ps1_guardrail.py --models qwen-voice qwen3.5:4b
python ps2_register.py  --models qwen-voice qwen3.5:4b
python ps3_tool_calls.py --models qwen-voice qwen3.5:4b
```

## Run Track 2 (PS-4, PS-5) — requires vLLM

```bash
# Start vLLM
pip install vllm
vllm serve Qwen/Qwen3.5-9B \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice

# Run
python run_all.py --models Qwen/Qwen3.5-9B --ps 4 5 --precisions Q4 Q8
```

## Results

All outputs land in `results/`:

| File | Contents |
|------|----------|
| `ps1_<model>.jsonl` | Per-case guardrail results |
| `ps1_summary.json` | Violation rates by model, category, language |
| `ps2_<model>.jsonl` | Register scores per scenario |
| `ps2_summary.json` | Avg scores + failure modes |
| `ps3_<model>.jsonl` | Tool-call accuracy per case |
| `ps3_summary.json` | Correct-tool rate, en-indic delta, error taxonomy |
| `ps4_<model>.json` | Latency curves (p50/p95/p99 TTFT) |
| `ps5_quantization_cliff.json` | Degradation across Q4/Q8/FP8/BF16 |
| `combined_results.json` | All summaries in one file |
| `audio/` | TTS audio samples (PS-2) |

Findings report: `report/findings.md`

## Models

| Model | Quantization | Role |
|-------|-------------|------|
| qwen3.5:9b (qwen-voice) | Q4 | Primary Track 1 |
| qwen3.5:4b | Q4 | Smallest viable |
| Qwen/Qwen3.5-9B | BF16 | Track 2 baseline |

Baseline API model: declare in submission (one hosted API of your choice).

## Limitations

- Q4 results are not production conclusions (see PS-5 for quantization cliff).
- LLM-as-judge validated against human labels on 20-case subset; kappa reported in `ps1_summary.json`.
- TTS uses `edge-tts` (hi-IN-SwaraNeural); production TTS engines may produce different artefacts.
- PS-4/PS-5 require Track 2 GPU access; Track 1 stubs run but measure Ollama (single-stream).

## License

Apache 2.0
