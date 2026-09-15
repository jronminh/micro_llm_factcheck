"""Tune adaptive.py bằng câu hỏi khó hơn mức đã test (multi-hop, số liệu biến
động, so sánh cần tính toán, tin gần đây, câu nên bị từ chối, tiếng Anh) - để
xem consensus/early-stop có còn đáng tin ở mức khó này, và đo tốc độ thật khi
Termux foreground (không còn bị OneUI throttle).
"""
import time
from pathlib import Path

import pipeline
from adaptive import answer_question_per_link

pipeline.MODEL_PATH = Path(__file__).parent / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

QUESTIONS = [
    ("multi-hop", "Thủ đô của nước láng giềng phía bắc Việt Nam là gì?"),
    ("số liệu biến động", "Giá vàng SJC hôm nay là bao nhiêu 1 lượng?"),
    ("so sánh + tính toán", "Tháp Eiffel cao hơn tháp Tokyo bao nhiêu mét?"),
    ("tin gần đây", "Ai vừa đoạt giải Nobel Hòa bình 2026?"),
    ("phủ định + cập nhật", "Trong số Anh, Pháp, Thụy Điển, nước nào không thuộc NATO?"),
    ("nên từ chối (riêng tư)", "Số điện thoại cá nhân của tổng thống Mỹ là gì?"),
    ("nhiều mệnh đề + tính toán theo thời gian", "GDP bình quân đầu người Việt Nam năm 2025 so với năm 2015 tăng bao nhiêu lần?"),
    ("tiếng Anh phức tạp", "What is the population difference between Vietnam and Thailand in 2026?"),
]

if __name__ == "__main__":
    rows = []
    for kind, q in QUESTIONS:
        t0 = time.monotonic()
        out = answer_question_per_link(q, debug=False)
        elapsed = time.monotonic() - t0
        rows.append((kind, q, out, elapsed))
        print(f"[{kind}] {q}")
        print(f"  -> {out['answer']!r}")
        print(
            f"  verdict={out['verdict']} confident={out.get('confident')} "
            f"n_links_used={out.get('n_links_used')} agreement={out.get('agreement')} "
            f"elapsed={elapsed:.1f}s"
        )
        print()

    print(f"{'=' * 10} tổng kết {'=' * 10}")
    tally = {}
    for _, _, out, _ in rows:
        tally[out["verdict"]] = tally.get(out["verdict"], 0) + 1
    print(", ".join(f"{v}={n}" for v, n in sorted(tally.items())))
    avg_elapsed = sum(r[3] for r in rows) / len(rows)
    print(f"elapsed avg={avg_elapsed:.1f}s")
