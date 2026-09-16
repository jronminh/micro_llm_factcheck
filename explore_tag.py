"""Khảo sát v2 cho việc tách câu hỏi multi-hop: thay vì bắt model TỰ SINH câu
hỏi con (mode "decompose" - dễ hallucinate, xem case "phía Tây" trong
explore_decompose.py), chỉ bắt model GẮN NHÃN các cụm có sẵn trong câu hỏi gốc
(chu_ngu/tu_noi/phu_ngu/phan_hoi), lấy nguyên văn không viết lại - gần với mode
"extract" (trích dẫn) vốn đáng tin hơn "synth" (tự sinh). Python tự ráp câu hỏi
con từ nhãn, dùng lại decompose._needs_lookup làm lưới an toàn.

So với decompose.py (regex thuần, chỉ bắt "của"/"hơn" tường minh): kỳ vọng
model bắt được cả trường hợp KHÔNG có từ nối tường minh (tiếng Việt hay lược bỏ
"của" trong cụm sở hữu, vd "Dân số thành phố lớn nhất Nhật Bản" - regex bó tay
vì không có literal "của" để khớp).
"""
import json
import re
from pathlib import Path

import llm
from decompose import _needs_lookup
from explore_decompose import QUESTIONS as BASE_QUESTIONS

llm.MODEL_PATH = Path(__file__).parent / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

EXTRA_QUESTIONS = [
    ("genitive ẩn, không có 'của' (regex bó tay)", "Dân số thành phố lớn nhất Nhật Bản là bao nhiêu?"),
    ("genitive có 'của' nhưng entity đã cụ thể (KHÔNG nên tách)", "Giá vàng của SJC hôm nay là bao nhiêu?"),
    ("genitive có 'của', entity đã cụ thể (control)", "Thủ đô của Việt Nam là gì?"),
]
QUESTIONS = BASE_QUESTIONS + EXTRA_QUESTIONS

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_tag(raw: str) -> dict | None:
    match = _JSON_RE.search(raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        keys = ("chu_ngu", "tu_noi", "phu_ngu", "phan_hoi")
        return data if all(k in data and isinstance(data[k], str) for k in keys) else None
    except json.JSONDecodeError:
        return None


def assemble(tag: dict) -> list[str] | None:
    chu_ngu, tu_noi, phu_ngu, phan_hoi = (tag[k].strip() for k in ("chu_ngu", "tu_noi", "phu_ngu", "phan_hoi"))
    if not phu_ngu or not _needs_lookup(phu_ngu):
        return None

    if tu_noi == "hơn":
        parts = chu_ngu.rsplit(" ", 1)
        if len(parts) != 2:
            return None
        entity_a, adjective = parts
        return [f"{entity_a} {adjective} {phan_hoi}?", f"{phu_ngu} {adjective} {phan_hoi}?"]

    connector = f" {tu_noi}" if tu_noi else ""
    return [phu_ngu, f"{chu_ngu}{connector} {{X}} {phan_hoi}".strip() + "?"]


if __name__ == "__main__":
    llm.ensure_server()
    tally = {"valid_json": 0, "invalid": 0}
    for kind, q in QUESTIONS:
        result = llm.run_model(q, mode="tag", temperature=0.0, n_predict=150)
        tag = parse_tag(result["answer"])
        tally["valid_json" if tag is not None else "invalid"] += 1
        print(f"[{kind}] {q}")
        print(f"  raw -> {result['answer']!r}")
        if tag is not None:
            steps = assemble(tag)
            tag_str = ", ".join(f"{k}={v!r}" for k, v in tag.items())
            print(f"  tag -> {tag_str}")
            print(f"  steps -> {steps if steps else '(giữ nguyên, không tách)'}")
        else:
            print("  tag -> KHÔNG PARSE ĐƯỢC JSON")
        print()

    print(f"{'=' * 10} tổng kết {'=' * 10}")
    print(f"valid_json={tally['valid_json']}/{len(QUESTIONS)}, invalid={tally['invalid']}/{len(QUESTIONS)}")
