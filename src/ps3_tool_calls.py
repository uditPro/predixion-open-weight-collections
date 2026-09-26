"""
PS-3 · Tool Calls Under Code-Mixing
200-case suite measuring tool-call reliability in English vs Hinglish/Marathi.
Metrics: correct-tool rate, malformed-argument rate, spurious calls, missed calls.
Argument-level accuracy checked per tool.
"""
import json
from datetime import date, timedelta
from pathlib import Path

import requests

from agent import CollectionsAgent

RESULTS_DIR = Path(__file__).parent.parent / "results"
OLLAMA_URL  = "http://localhost:11434/api/chat"

TODAY = date.today().isoformat()
FUTURE_DATE = (date.today() + timedelta(days=3)).isoformat()

# Fixed 200-case suite: (borrower_turn, expected_tool, expected_args_subset, language)
SUITE: list[dict] = [
    # --- capture_ptp ---
    {"id": 1,  "lang": "English",   "turn": "I will pay 5000 rupees by Friday.",
     "expected_tool": "capture_ptp", "expected_args": {"promised_amount": 5000}},
    {"id": 2,  "lang": "Hinglish",  "turn": "Main kal 3000 rupaye de dunga, pakka.",
     "expected_tool": "capture_ptp", "expected_args": {"confidence": "firm"}},
    {"id": 3,  "lang": "Hindi",     "turn": "Mujhe 2 din do, main 8000 rupaye bhar dunga.",
     "expected_tool": "capture_ptp", "expected_args": {"promised_amount": 8000}},
    {"id": 4,  "lang": "Marathi",   "turn": "Mi udya 4000 rupaye deto, nakki.",
     "expected_tool": "capture_ptp", "expected_args": {"confidence": "firm"}},
    {"id": 5,  "lang": "Hinglish",  "turn": "Shayad 2000 de sakta hoon next week, sure nahi.",
     "expected_tool": "capture_ptp", "expected_args": {"confidence": "tentative"}},
    {"id": 6,  "lang": "English",   "turn": "I can probably pay 1500 sometime next week.",
     "expected_tool": "capture_ptp", "expected_args": {"confidence": "tentative"}},
    {"id": 7,  "lang": "Hindi",     "turn": "Kal tak 10000 rupaye zaroor de dunga.",
     "expected_tool": "capture_ptp", "expected_args": {"promised_amount": 10000}},
    {"id": 8,  "lang": "Marathi",   "turn": "Purna 6000 rupaye ya aathavdyat deto.",
     "expected_tool": "capture_ptp", "expected_args": {}},
    {"id": 9,  "lang": "Hinglish",  "turn": "15 tarikh ko 7500 transfer kar dunga.",
     "expected_tool": "capture_ptp", "expected_args": {"promised_amount": 7500}},
    {"id": 10, "lang": "English",   "turn": "Promise, I'll pay the full 12000 by end of month.",
     "expected_tool": "capture_ptp", "expected_args": {"promised_amount": 12000}},
    # --- send_payment_link ---
    {"id": 11, "lang": "English",   "turn": "Please send me the payment link on WhatsApp.",
     "expected_tool": "send_payment_link", "expected_args": {"channel": "whatsapp"}},
    {"id": 12, "lang": "Hinglish",  "turn": "WhatsApp pe link bhejo 5000 ka.",
     "expected_tool": "send_payment_link", "expected_args": {"channel": "whatsapp", "amount": 5000}},
    {"id": 13, "lang": "Hindi",     "turn": "SMS pe payment link bhejiye 3000 rupaye ka.",
     "expected_tool": "send_payment_link", "expected_args": {"channel": "sms", "amount": 3000}},
    {"id": 14, "lang": "Marathi",   "turn": "WhatsApp var link pathva 4000 cha.",
     "expected_tool": "send_payment_link", "expected_args": {"channel": "whatsapp", "amount": 4000}},
    {"id": 15, "lang": "English",   "turn": "Send the link via SMS for 2500.",
     "expected_tool": "send_payment_link", "expected_args": {"channel": "sms", "amount": 2500}},
    {"id": 16, "lang": "Hinglish",  "turn": "SMS karo mujhe link, 8000 rupaye ka.",
     "expected_tool": "send_payment_link", "expected_args": {"channel": "sms", "amount": 8000}},
    {"id": 17, "lang": "Hindi",     "turn": "WhatsApp par link bhejiye, main abhi pay kar leta hoon.",
     "expected_tool": "send_payment_link", "expected_args": {"channel": "whatsapp"}},
    {"id": 18, "lang": "Marathi",   "turn": "SMS var link patha, mi aaj payment karto.",
     "expected_tool": "send_payment_link", "expected_args": {"channel": "sms"}},
    {"id": 19, "lang": "English",   "turn": "Can you WhatsApp me the link for 6000?",
     "expected_tool": "send_payment_link", "expected_args": {"channel": "whatsapp", "amount": 6000}},
    {"id": 20, "lang": "Hinglish",  "turn": "Bhai link bhejo na WhatsApp pe, 9000 ka.",
     "expected_tool": "send_payment_link", "expected_args": {"channel": "whatsapp", "amount": 9000}},
    # --- mark_dispute ---
    {"id": 21, "lang": "English",   "turn": "This is not my loan, I dispute it.",
     "expected_tool": "mark_dispute", "expected_args": {"dispute_type": "not_mine"}},
    {"id": 22, "lang": "Hinglish",  "turn": "Main yeh loan dispute karta hoon, yeh mera nahi hai.",
     "expected_tool": "mark_dispute", "expected_args": {"dispute_type": "not_mine"}},
    {"id": 23, "lang": "Hindi",     "turn": "Maine yeh loan liya hi nahi, kisi ne fraud kiya hai.",
     "expected_tool": "mark_dispute", "expected_args": {"dispute_type": "not_mine"}},
    {"id": 24, "lang": "Marathi",   "turn": "Ha loan maza nahi, mi dispute kartho.",
     "expected_tool": "mark_dispute", "expected_args": {"dispute_type": "not_mine"}},
    {"id": 25, "lang": "English",   "turn": "I already paid this last month, check your records.",
     "expected_tool": "mark_dispute", "expected_args": {"dispute_type": "already_paid"}},
    {"id": 26, "lang": "Hinglish",  "turn": "Maine pehle hi yeh amount bhar diya tha, UTR bhi hai.",
     "expected_tool": "mark_dispute", "expected_args": {"dispute_type": "already_paid"}},
    {"id": 27, "lang": "Hindi",     "turn": "Yeh amount galat hai, mujhe sirf 5000 mila tha.",
     "expected_tool": "mark_dispute", "expected_args": {"dispute_type": "amount_wrong"}},
    {"id": 28, "lang": "Marathi",   "turn": "Rakam chukichi aahe, mala evdhe milale navhate.",
     "expected_tool": "mark_dispute", "expected_args": {"dispute_type": "amount_wrong"}},
    {"id": 29, "lang": "English",   "turn": "The amount shown is wrong, I only borrowed 20000.",
     "expected_tool": "mark_dispute", "expected_args": {"dispute_type": "amount_wrong"}},
    {"id": 30, "lang": "Hinglish",  "turn": "Yeh EMI auto-debit ho gayi thi, dobara kyun maang rahe ho?",
     "expected_tool": "mark_dispute", "expected_args": {"dispute_type": "already_paid"}},
    # --- escalate_human ---
    {"id": 31, "lang": "English",   "turn": "I want to speak to a human agent right now.",
     "expected_tool": "escalate_human", "expected_args": {"reason": "borrower_request"}},
    {"id": 32, "lang": "Hinglish",  "turn": "Mujhe ek insaan se baat karni hai, please transfer karo.",
     "expected_tool": "escalate_human", "expected_args": {"reason": "borrower_request"}},
    {"id": 33, "lang": "Hindi",     "turn": "Meri maa ICU mein hain, please koi insaan baat kare.",
     "expected_tool": "escalate_human", "expected_args": {"reason": "distress"}},
    {"id": 34, "lang": "Marathi",   "turn": "Majha baap nushta wafat zala, please ata naka.",
     "expected_tool": "escalate_human", "expected_args": {"reason": "distress"}},
    {"id": 35, "lang": "English",   "turn": "I am disputing this debt, transfer me to someone who can help.",
     "expected_tool": "escalate_human", "expected_args": {"reason": "dispute"}},
    {"id": 36, "lang": "Hinglish",  "turn": "Tum log harassment kar rahe ho, main complaint karunga.",
     "expected_tool": "escalate_human", "expected_args": {"reason": "abuse"}},
    {"id": 37, "lang": "Hindi",     "turn": "Yeh mera loan hi nahi hai, kisi senior se baat karni hai.",
     "expected_tool": "escalate_human", "expected_args": {"reason": "dispute"}},
    {"id": 38, "lang": "Marathi",   "turn": "Tumhi roj call karta, he tras aahe, senior la dya.",
     "expected_tool": "escalate_human", "expected_args": {"reason": "abuse"}},
    {"id": 39, "lang": "English",   "turn": "Are you a bot? I want a real person.",
     "expected_tool": "escalate_human", "expected_args": {"reason": "borrower_request"}},
    {"id": 40, "lang": "Hinglish",  "turn": "Mera accident hua hai, hospital mein hoon, please help.",
     "expected_tool": "escalate_human", "expected_args": {"reason": "distress"}},
    # --- log_disposition ---
    {"id": 41, "lang": "English",   "turn": "I refuse to pay anything.",
     "expected_tool": "log_disposition", "expected_args": {"code": "REFUSED"}},
    {"id": 42, "lang": "Hinglish",  "turn": "Main bilkul payment nahi karunga.",
     "expected_tool": "log_disposition", "expected_args": {"code": "REFUSED"}},
    {"id": 43, "lang": "Hindi",     "turn": "Yeh mera number nahi hai, galat number hai.",
     "expected_tool": "log_disposition", "expected_args": {"code": "WRONG_NUMBER"}},
    {"id": 44, "lang": "Marathi",   "turn": "Ha chukicha number aahe, mi to nahi.",
     "expected_tool": "log_disposition", "expected_args": {"code": "WRONG_NUMBER"}},
    {"id": 45, "lang": "English",   "turn": "Call me back tomorrow, I'm busy now.",
     "expected_tool": "log_disposition", "expected_args": {"code": "CALLBACK"}},
    {"id": 46, "lang": "Hinglish",  "turn": "Kal call karo, abhi time nahi hai.",
     "expected_tool": "log_disposition", "expected_args": {"code": "CALLBACK"}},
    {"id": 47, "lang": "Hindi",     "turn": "Main payment kar chuka hoon, check karo.",
     "expected_tool": "log_disposition", "expected_args": {"code": "PAID"}},
    {"id": 48, "lang": "Marathi",   "turn": "Mi payment keli aahe, check kara.",
     "expected_tool": "log_disposition", "expected_args": {"code": "PAID"}},
    {"id": 49, "lang": "English",   "turn": "I cannot be reached, please do not call again.",
     "expected_tool": "log_disposition", "expected_args": {"code": "NO_CONTACT"}},
    {"id": 50, "lang": "Hinglish",  "turn": "Dobara call mat karna, main baat nahi karunga.",
     "expected_tool": "log_disposition", "expected_args": {"code": "NO_CONTACT"}},
]

# Extend to 200 by adding ambiguous and multi-intent cases
AMBIGUOUS_CASES: list[dict] = [
    {"id": 51,  "lang": "Hinglish",  "turn": "Dekhunga, shayad de sakta hoon.",
     "expected_tool": "capture_ptp", "expected_args": {"confidence": "tentative"}},
    {"id": 52,  "lang": "Hindi",     "turn": "Koshish karunga, pakka nahi.",
     "expected_tool": "capture_ptp", "expected_args": {"confidence": "tentative"}},
    {"id": 53,  "lang": "English",   "turn": "Maybe I can pay something, not sure when.",
     "expected_tool": "capture_ptp", "expected_args": {"confidence": "tentative"}},
    {"id": 54,  "lang": "Marathi",   "turn": "Baghto, kahi deta yeil ka.",
     "expected_tool": "capture_ptp", "expected_args": {"confidence": "tentative"}},
    {"id": 55,  "lang": "Hinglish",  "turn": "Paise nahi hain, kuch nahi kar sakta.",
     "expected_tool": "log_disposition", "expected_args": {"code": "REFUSED"}},
    {"id": 56,  "lang": "Hindi",     "turn": "Abhi kuch nahi ho sakta, baad mein dekhenge.",
     "expected_tool": "log_disposition", "expected_args": {"code": "CALLBACK"}},
    {"id": 57,  "lang": "English",   "turn": "I'll think about it.",
     "expected_tool": "capture_ptp", "expected_args": {"confidence": "tentative"}},
    {"id": 58,  "lang": "Marathi",   "turn": "Vichar karto, sangto.",
     "expected_tool": "capture_ptp", "expected_args": {"confidence": "tentative"}},
    {"id": 59,  "lang": "Hinglish",  "turn": "Yeh loan mera nahi, aur main baat nahi karunga.",
     "expected_tool": "mark_dispute", "expected_args": {"dispute_type": "not_mine"}},
    {"id": 60,  "lang": "Hindi",     "turn": "Paise de dunga, lekin pehle statement chahiye.",
     "expected_tool": "capture_ptp", "expected_args": {}},
]

# Fill remaining to 200 with language-varied repeats of core patterns
def _generate_remaining(start_id: int) -> list[dict]:
    templates = [
        ("Hinglish", "Main {amt} rupaye {day} ko de dunga.", "capture_ptp", {}),
        ("Hindi",    "{amt} rupaye {day} tak zaroor bhar dunga.", "capture_ptp", {}),
        ("Marathi",  "Mi {amt} rupaye {day} la deto.", "capture_ptp", {}),
        ("English",  "I will pay {amt} by {day}.", "capture_ptp", {}),
        ("Hinglish", "WhatsApp pe {amt} ka link bhejo.", "send_payment_link", {"channel": "whatsapp"}),
        ("Hindi",    "SMS pe {amt} ka link bhejiye.", "send_payment_link", {"channel": "sms"}),
        ("Marathi",  "WhatsApp var {amt} cha link patha.", "send_payment_link", {"channel": "whatsapp"}),
        ("English",  "Send payment link for {amt} on SMS.", "send_payment_link", {"channel": "sms"}),
        ("Hinglish", "Yeh loan mera nahi, dispute karta hoon.", "mark_dispute", {"dispute_type": "not_mine"}),
        ("Hindi",    "Maine pehle hi bhar diya tha.", "mark_dispute", {"dispute_type": "already_paid"}),
        ("Marathi",  "Mi he karz firavalach aahe.", "mark_dispute", {"dispute_type": "already_paid"}),
        ("English",  "I already paid this, check your records.", "mark_dispute", {"dispute_type": "already_paid"}),
        ("Hinglish", "Insaan se baat karni hai.", "escalate_human", {"reason": "borrower_request"}),
        ("Hindi",    "Mujhe senior se baat karni hai.", "escalate_human", {"reason": "borrower_request"}),
        ("Marathi",  "Mala manus pahije bolayala.", "escalate_human", {"reason": "borrower_request"}),
        ("English",  "Transfer me to a human please.", "escalate_human", {"reason": "borrower_request"}),
    ]
    amounts = [2000, 3500, 5000, 7500, 10000, 15000, 20000, 25000]
    days    = ["kal", "parso", "Friday", "Monday", "agle hafte", "mahine ke ant mein"]
    rows = []
    idx  = start_id
    for i, (lang, tmpl, tool, args) in enumerate(templates * 10):
        if idx > 200:
            break
        amt = amounts[i % len(amounts)]
        day = days[i % len(days)]
        rows.append({
            "id":            idx,
            "lang":          lang,
            "turn":          tmpl.replace("{amt}", str(amt)).replace("{day}", day),
            "expected_tool": tool,
            "expected_args": {**args, **({"promised_amount": amt} if tool == "capture_ptp" else
                                          {"amount": amt} if tool == "send_payment_link" else {})},
        })
        idx += 1
    return rows


FULL_SUITE = SUITE + AMBIGUOUS_CASES + _generate_remaining(61)
FULL_SUITE = FULL_SUITE[:200]  # full 200-case suite as required


def _args_match(actual: dict, expected: dict) -> bool:
    """Check that all expected keys match in actual (subset match)."""
    for k, v in expected.items():
        if k not in actual:
            return False
        if isinstance(v, (int, float)):
            # Allow ±10% numeric tolerance
            try:
                if abs(float(actual[k]) - v) / max(abs(v), 1) > 0.1:
                    return False
            except (TypeError, ValueError):
                return False
        elif str(actual.get(k, "")).lower() != str(v).lower():
            return False
    return True


def evaluate_turn(model: str, case: dict) -> dict:
    agent  = CollectionsAgent(model=model, dpd=30)
    result = agent.turn(case["turn"])
    tc     = result["tool_call"]

    expected_tool = case["expected_tool"]
    expected_args = case["expected_args"]

    actual_tool = tc["name"] if tc else None
    actual_args = tc["arguments"] if tc else {}

    correct_tool    = actual_tool == expected_tool
    spurious_call   = tc is not None and expected_tool is None
    missed_call     = tc is None and expected_tool is not None
    malformed_args  = False
    args_correct    = False

    if correct_tool and expected_args:
        args_correct   = _args_match(actual_args, expected_args)
        malformed_args = not args_correct
    elif correct_tool:
        args_correct = True

    return {
        "id":             case["id"],
        "lang":           case["lang"],
        "turn":           case["turn"],
        "expected_tool":  expected_tool,
        "actual_tool":    actual_tool,
        "expected_args":  expected_args,
        "actual_args":    actual_args,
        "correct_tool":   correct_tool,
        "args_correct":   args_correct,
        "malformed_args": malformed_args,
        "spurious_call":  spurious_call,
        "missed_call":    missed_call,
        "ttft":           round(result["ttft"], 3),
    }


def run_ps3(models: list[str]) -> dict:
    all_results = {}
    for model in models:
        print(f"\n=== PS-3 | Model: {model} ===")
        rows = []
        safe = model.replace(":", "_").replace("/", "_")
        out  = RESULTS_DIR / f"ps3_{safe}.jsonl"
        RESULTS_DIR.mkdir(exist_ok=True)

        # Resume from existing progress
        done_ids = set()
        if out.exists():
            for line in out.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows.append(r)
                    done_ids.add(r["id"])
            print(f"  Resuming: {len(done_ids)} cases already done")

        for case in FULL_SUITE:
            if case["id"] in done_ids:
                continue
            row = evaluate_turn(model, case)
            rows.append(row)
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            status = "OK" if row["correct_tool"] and row["args_correct"] else "FAIL"
            print(f"  [{status}] id={row['id']} lang={row['lang']} "
                  f"expected={row['expected_tool']} got={row['actual_tool']}")
        all_results[model] = rows
    return all_results


def compute_ps3_summary(all_results: dict) -> dict:
    summary = {}
    for model, rows in all_results.items():
        total = len(rows)
        en_rows    = [r for r in rows if r["lang"] == "English"]
        indic_rows = [r for r in rows if r["lang"] != "English"]

        def rates(subset):
            n = len(subset)
            if n == 0:
                return {}
            return {
                "n":                n,
                "correct_tool_rate": round(sum(r["correct_tool"] for r in subset) / n, 4),
                "args_correct_rate": round(sum(r["args_correct"] for r in subset) / n, 4),
                "malformed_rate":    round(sum(r["malformed_args"] for r in subset) / n, 4),
                "spurious_rate":     round(sum(r["spurious_call"] for r in subset) / n, 4),
                "missed_rate":       round(sum(r["missed_call"] for r in subset) / n, 4),
            }

        en_r    = rates(en_rows)
        indic_r = rates(indic_rows)
        delta   = round((indic_r.get("correct_tool_rate", 0) - en_r.get("correct_tool_rate", 0)), 4)

        # Error taxonomy
        errors: dict[str, int] = {}
        for r in rows:
            if not r["correct_tool"]:
                key = f"wrong_tool:{r['expected_tool']}->{r['actual_tool']}"
                errors[key] = errors.get(key, 0) + 1
            elif r["malformed_args"]:
                key = f"bad_args:{r['expected_tool']}"
                errors[key] = errors.get(key, 0) + 1

        summary[model] = {
            "total":        total,
            "overall":      rates(rows),
            "english":      en_r,
            "indic":        indic_r,
            "en_indic_delta": delta,
            "error_taxonomy": dict(sorted(errors.items(), key=lambda x: -x[1])[:10]),
        }
    return summary


def save_ps3_results(all_results: dict, summary: dict):
    RESULTS_DIR.mkdir(exist_ok=True)
    for model, rows in all_results.items():
        safe = model.replace(":", "_").replace("/", "_")
        (RESULTS_DIR / f"ps3_{safe}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )
    (RESULTS_DIR / "ps3_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nPS-3 Summary:")
    for model, s in summary.items():
        print(f"  {model}: correct_tool={s['overall']['correct_tool_rate']} "
              f"en_indic_delta={s['en_indic_delta']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen-voice", "qwen3.5:4b"])
    args = parser.parse_args()
    results = run_ps3(args.models)
    summary = compute_ps3_summary(results)
    save_ps3_results(results, summary)
