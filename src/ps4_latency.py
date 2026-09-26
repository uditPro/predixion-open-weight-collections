"""
PS-4 · p95 Under Month-End Load  (Track 2 — vLLM)
Concurrency sweep: 1, 10, 25, 50, 100.
Reports p50/p95/p99 TTFT + inter-token latency + cost curve.
Requires vLLM running at VLLM_URL.
"""
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp

RESULTS_DIR = Path(__file__).parent.parent / "results"
VLLM_URL    = "http://localhost:8000/v1/chat/completions"

# Cost assumptions (ap-south-1 g5.xlarge ~$1.006/hr, ~1 GPU)
COST_PER_HOUR_USD = 1.006
AVG_CALL_TOKENS   = 512   # input + output per turn

CONCURRENCY_LEVELS = [1, 10, 25, 50, 100]

SAMPLE_MESSAGES = [
    [{"role": "system", "content": "You are a collections agent. Keep replies short."},
     {"role": "user",   "content": "Main kal 5000 rupaye de dunga, pakka."}],
    [{"role": "system", "content": "You are a collections agent. Keep replies short."},
     {"role": "user",   "content": "Yeh loan mera nahi hai, main dispute karta hoon."}],
    [{"role": "system", "content": "You are a collections agent. Keep replies short."},
     {"role": "user",   "content": "WhatsApp pe payment link bhejo 3000 ka."}],
    [{"role": "system", "content": "You are a collections agent. Keep replies short."},
     {"role": "user",   "content": "Mujhe insaan se baat karni hai please."}],
]


async def single_request(session: aiohttp.ClientSession, model: str, messages: list) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 128,
        "temperature": 0.3,
        "stream": True,
    }
    t_start = time.perf_counter()
    ttft    = None
    tokens  = 0
    t_last  = t_start

    try:
        async with session.post(VLLM_URL, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            async for line in resp.content:
                line = line.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        if ttft is None:
                            ttft = time.perf_counter() - t_start
                        tokens += 1
                        t_last  = time.perf_counter()
                except (json.JSONDecodeError, KeyError):
                    pass
    except Exception as e:
        return {"ttft": None, "total_time": None, "tokens": 0, "error": str(e)}

    total_time = time.perf_counter() - t_start
    itl = (t_last - (t_start + (ttft or 0))) / max(tokens - 1, 1)
    return {"ttft": ttft, "total_time": total_time, "tokens": tokens, "itl": itl}


async def run_concurrency_level(model: str, concurrency: int, n_requests: int = 50) -> dict:
    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(messages):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                return await single_request(session, model, messages)

    tasks = [bounded(SAMPLE_MESSAGES[i % len(SAMPLE_MESSAGES)]) for i in range(n_requests)]
    raw   = await asyncio.gather(*tasks)

    ttfts = [r["ttft"] for r in raw if r.get("ttft") is not None]
    itls  = [r["itl"]  for r in raw if r.get("itl")  is not None]

    if not ttfts:
        return {"concurrency": concurrency, "error": "all requests failed"}

    ttfts.sort()
    return {
        "concurrency": concurrency,
        "n_requests":  n_requests,
        "n_success":   len(ttfts),
        "ttft_p50":    round(statistics.median(ttfts), 3),
        "ttft_p95":    round(ttfts[int(len(ttfts) * 0.95)], 3),
        "ttft_p99":    round(ttfts[int(len(ttfts) * 0.99)], 3),
        "ttft_mean":   round(statistics.mean(ttfts), 3),
        "itl_mean":    round(statistics.mean(itls), 3) if itls else None,
    }


def cost_per_1000_calls(throughput_rps: float) -> float:
    """Derive cost per 1000 calls from measured throughput."""
    calls_per_hour = throughput_rps * 3600
    return round((COST_PER_HOUR_USD / calls_per_hour) * 1000, 4) if calls_per_hour > 0 else 0


def run_ps4(models: list[str]):
    RESULTS_DIR.mkdir(exist_ok=True)
    all_results = {}

    for model in models:
        print(f"\n=== PS-4 | Model: {model} ===")
        model_rows = []

        for c in CONCURRENCY_LEVELS:
            print(f"  Concurrency={c} ...", end=" ", flush=True)
            row = asyncio.run(run_concurrency_level(model, c))
            # Throughput = n_success / (p95 * n_success / c) approx
            if "ttft_p95" in row:
                tput = row["n_success"] / (row["ttft_p95"] * row["n_success"] / c)
                row["cost_per_1000_calls"] = cost_per_1000_calls(tput)
                row["throughput_rps"]      = round(tput, 2)
                # Saturation: p95 > 3s is noticeable pause
                row["saturated"] = row["ttft_p95"] > 3.0
            model_rows.append(row)
            print(f"p95={row.get('ttft_p95')}s saturated={row.get('saturated')}")

        all_results[model] = model_rows

        out = RESULTS_DIR / f"ps4_{model.replace(':', '_').replace('/', '_')}.json"
        out.write_text(json.dumps(model_rows, indent=2))
        print(f"  Saved {out}")

    (RESULTS_DIR / "ps4_summary.json").write_text(json.dumps(all_results, indent=2))
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen3.5-9B"])
    args = parser.parse_args()
    run_ps4(args.models)
