"""Test model 0.5B với nhiều dạng câu hỏi khác nhau, ở 2 mode:
"synth" (tổng hợp, được ghép câu) và "extract" (chỉ trích nguyên văn, ép
hoạt động như một công cụ search hơn là một model đang suy luận).
"""
from pathlib import Path

import pipeline

pipeline.MODEL_PATH = Path(__file__).parent / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

QUESTIONS = [
    ("câu hỏi đầy đủ", "Chiều cao núi Everest là bao nhiêu?"),
    ("có/không", "Việt Nam có biên giới với Trung Quốc không?"),
    ("kiểu từ khóa (search bar)", "dân số hà nội 2026"),
    ("so sánh", "Ai cao hơn, tháp Eiffel hay tháp Tokyo?"),
    ("liệt kê", "Các quốc gia tổ chức World Cup 2026 là những nước nào?"),
    ("mở/rộng", "Tình hình kinh tế Việt Nam năm 2026 thế nào?"),
    ("ngoài phạm vi (kỳ vọng từ chối)", "Nhiệt độ trên Sao Hỏa hôm nay là bao nhiêu độ?"),
    ("định nghĩa ngắn", "Blockchain là gì?"),
]

if __name__ == "__main__":
    for mode in ("synth", "extract"):
        print(f"\n{'=' * 10} mode={mode} {'=' * 10}")
        for kind, q in QUESTIONS:
            out = pipeline.answer_question(q, mode=mode)
            print(f"[{kind}] {q}")
            print(f"  -> {out['answer']}")
            print(f"  gen_tps={out['gen_tps']:.1f}")
            print()
