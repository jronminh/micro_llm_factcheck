"""So sánh performance giữa các bản model (1.5B vs 0.5B) trên cùng bộ câu hỏi.

Chạy tuần tự (không đồng thời) để tránh 2 model cùng chiếm RAM: đổi
MODEL_PATH rồi gọi ensure_server() - nó tự phát hiện server đang chạy
model khác và khởi động lại đúng model trước khi chạy hết bộ câu hỏi.
"""
import time
from pathlib import Path

import llm

MODELS_DIR = Path(__file__).parent / "models"
MODELS = {
    "1.5B": MODELS_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "0.5B": MODELS_DIR / "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    # Khảo sát model khác ngoài Qwen2.5, xem notes/micro-llm-alternatives-to-qwen.md.
    "qwen3-0.6b": MODELS_DIR / "qwen3-0.6b-q4_k_m.gguf",
    "gemma3-270m": MODELS_DIR / "gemma-3-270m-it-q4_k_m.gguf",
}

QUESTIONS = [
    "Chieu cao nui Everest la bao nhieu?",
    "Ai la tong thong My hien tai?",
    "Python la ngon ngu lap trinh gi?",
]


def run_for_model(label: str, model_path: Path) -> list[dict]:
    llm.MODEL_PATH = model_path
    llm.ensure_server()

    rows = []
    for q in QUESTIONS:
        t0 = time.monotonic()
        out = llm.answer_question(q)
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
