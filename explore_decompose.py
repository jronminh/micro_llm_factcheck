"""Khảo sát: model nhỏ có phân rã được câu hỏi multi-hop thành câu hỏi con để
search riêng từng bước không - trước khi đầu tư viết cơ chế chain (search bước
1 -> lấy answer -> nhét vào bước 2 -> search tiếp).

Chỉ test bước phân rã (model -> JSON câu hỏi con), CHƯA chạy search/extract
cho từng bước con - mode="decompose" không nhận snippet, chỉ nhận câu hỏi gốc.
Câu hỏi control (đã đơn giản) phải KHÔNG bị tách, nếu không model sẽ tách cả
câu đơn giản thành nhiều bước search thừa, tốn thời gian mà không lợi gì.
"""
import json
import re
import time
from pathlib import Path

import llm

llm.MODEL_PATH = Path(__file__).parent / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

QUESTIONS = [
    ("multi-hop (case đã biết sai)", "Thủ đô của nước láng giềng phía bắc Việt Nam là gì?"),
    ("multi-hop", "Cầu thủ ghi bàn nhiều nhất World Cup 2026 chơi cho câu lạc bộ nào?"),
    ("so sánh 2 số liệu theo thời gian", "GDP bình quân đầu người Việt Nam năm 2025 so với năm 2015 tăng bao nhiêu lần?"),
    ("so sánh 2 thực thể", "Tháp Eiffel cao hơn tháp Tokyo Skytree bao nhiêu mét?"),
    ("đơn giản (control, KHÔNG nên tách)", "Thủ đô Việt Nam là gì?"),
    ("đơn giản, kiểu từ khóa (control)", "dân số hà nội 2026"),
    ("multi-hop tiếng Anh", "What is the capital of the country that borders Vietnam to the north?"),
]

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_steps(raw: str) -> list[str] | None:
    match = _JSON_RE.search(raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        steps = data.get("steps")
        return steps if isinstance(steps, list) and all(isinstance(s, str) for s in steps) else None
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    llm.ensure_server()
    tally = {"valid_json": 0, "invalid": 0}
    for kind, q in QUESTIONS:
        t0 = time.monotonic()
        result = llm.run_model(q, mode="decompose", temperature=0.0, n_predict=150)
        elapsed = time.monotonic() - t0
        steps = parse_steps(result["answer"])
        tally["valid_json" if steps is not None else "invalid"] += 1
        print(f"[{kind}] {q}")
        print(f"  raw -> {result['answer']!r}")
        if steps is not None:
            print(f"  steps ({len(steps)}) -> {steps}")
        else:
            print("  steps -> KHÔNG PARSE ĐƯỢC JSON")
        print(f"  gen_tps={result['gen_tps']:.1f} elapsed={elapsed:.1f}s")
        print()

    print(f"{'=' * 10} tổng kết {'=' * 10}")
    print(f"valid_json={tally['valid_json']}/{len(QUESTIONS)}, invalid={tally['invalid']}/{len(QUESTIONS)}")
