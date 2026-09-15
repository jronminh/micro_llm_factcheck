"""Đo thời gian từng bước (search, model) và ghi kết quả vào benchmark/results.csv."""
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pipeline import build_user_prompt, run_model
from search import search

RESULTS_CSV = Path(__file__).parent / "benchmark" / "results.csv"
FIELDS = [
    "timestamp", "question", "search_s", "model_s", "total_s",
    "prompt_tps", "gen_tps", "n_snippets", "answer_preview",
]


def run_benchmark(question: str, n_results: int = 5) -> dict:
    t0 = time.monotonic()
    snippets = search(question, n=n_results)
    t1 = time.monotonic()

    user_prompt = build_user_prompt(question, snippets)
    model_result = run_model(user_prompt)
    t2 = time.monotonic()

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question,
        "search_s": round(t1 - t0, 2),
        "model_s": round(t2 - t1, 2),
        "total_s": round(t2 - t0, 2),
        "prompt_tps": model_result["prompt_tps"],
        "gen_tps": model_result["gen_tps"],
        "n_snippets": len(snippets),
        "answer_preview": model_result["answer"][:120].replace("\n", " "),
    }
    return row


def append_result(row: dict) -> None:
    RESULTS_CSV.parent.mkdir(exist_ok=True)
    is_new = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Thủ đô Việt Nam là gì?"
    row = run_benchmark(q)
    for k, v in row.items():
        print(f"{k}: {v}")
    append_result(row)
    print(f"\nĐã ghi vào {RESULTS_CSV}")
