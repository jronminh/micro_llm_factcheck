"""So sánh performance giữa các bản model (1.5B vs 0.5B) trên cùng bộ câu hỏi.

Chạy tuần tự (không đồng thời) để tránh 2 model cùng chiếm RAM: dừng
server hiện tại, đổi MODEL_PATH, khởi động lại, chạy hết bộ câu hỏi, rồi
mới chuyển model kế tiếp.
"""
import subprocess
import time
from pathlib import Path

import pipeline

MODELS_DIR = Path(__file__).parent / "models"
MODELS = {
    "1.5B": MODELS_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "0.5B": MODELS_DIR / "qwen2.5-0.5b-instruct-q4_k_m.gguf",
}

QUESTIONS = [
    "Chieu cao nui Everest la bao nhieu?",
    "Ai la tong thong My hien tai?",
    "Python la ngon ngu lap trinh gi?",
]


def stop_server() -> None:
    # -x (khớp đúng tên process) thay vì -f: -f khớp cả cmdline của process
    # gọi pkill, có thể tự kill nhầm shell đang chạy nó.
    subprocess.run(["pkill", "-x", "llama-server"], check=False)
    time.sleep(1)


def run_for_model(label: str, model_path: Path) -> list[dict]:
    stop_server()
    pipeline.MODEL_PATH = model_path
    pipeline.ensure_server()

    rows = []
    for q in QUESTIONS:
        t0 = time.monotonic()
        out = pipeline.answer_question(q)
        elapsed = time.monotonic() - t0
        rows.append({
            "model": label,
            "question": q,
            "total_s": round(elapsed, 2),
            "gen_tps": out["gen_tps"],
            "answer_preview": out["answer"][:100].replace("\n", " "),
        })
    return rows


if __name__ == "__main__":
    all_rows = []
    for label, path in MODELS.items():
        if not path.exists():
            print(f"Bỏ qua {label}: chưa có file {path}")
            continue
        print(f"== Chạy model {label} ==")
        all_rows.extend(run_for_model(label, path))

    print(f"\n{'model':<6} {'total_s':>8} {'gen_tps':>8}  question / answer")
    for r in all_rows:
        print(f"{r['model']:<6} {r['total_s']:>8} {r['gen_tps']:>8}  {r['question']} -> {r['answer_preview']}")
