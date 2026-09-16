"""Chạy search+extract theo từng câu hỏi con đã tách bởi decompose.py, nối kết
quả bước trước vào câu hỏi bước sau (bridge) hoặc chạy song song (comparison).

CHƯA làm bước tổng hợp câu trả lời cuối cho câu hỏi gốc (vd rút "Bắc Kinh" ra
từ answer thô của bước 2, hay tính tỉ lệ GDP từ 2 số của comparison) - đó là
việc của "đầu ra", chưa cần ở giai đoạn này. Hàm ở đây chỉ trả về chuỗi
câu hỏi con -> câu trả lời con thô, để nhìn được search+chain có hoạt động
đúng hay không trước khi nghĩ tiếp bước tổng hợp.

Mỗi bước con vẫn dùng cơ chế đồng thuận đa link (không phải 1 lần gọi model)
- answer_question_per_link() (adaptive.py) cho bước cần câu trả lời đầy đủ,
answer_entity_per_link() (adaptive_entity.py, fork riêng) cho bước cần đúng 1
tên riêng ngắn (bridge entity) - không viết lại logic consensus/bias.
"""
from adaptive import answer_question_per_link
from adaptive_entity import answer_entity_per_link
from decompose import decompose

_NUMBER_MARKERS = ("bao nhiêu", "how much", "how many")


def _expects_number(query: str) -> bool:
    lower = query.lower()
    return any(m in lower for m in _NUMBER_MARKERS)


def answer_chain(question: str, **adaptive_kwargs) -> dict:
    debug = adaptive_kwargs.get("debug", False)
    steps = decompose(question)
    if debug:
        print(f"[debug][chain] decompose({question!r}) -> {steps!r}", flush=True)

    if len(steps) == 1:
        if debug:
            print("[debug][chain] type=none (không tách được), chạy như câu hỏi thường", flush=True)
        return {"question": question, "type": "none", "steps": [
            {"query": question, "result": answer_question_per_link(question, **adaptive_kwargs)},
        ]}

    if "{X}" in steps[1]:
        if debug:
            print(f"[debug][chain] type=bridge, bước 1: {steps[0]!r} (answer_entity_per_link)", flush=True)
        # Bridge entity luôn là 1 tên riêng ngắn (tên nước/thành phố/người),
        # khác với câu trả lời fact-check nói chung mà adaptive.py vốn được
        # tune cho - dùng adaptive_entity.py (fork riêng, prompt "extract_entity"
        # ép trả lời đúng 1 tên riêng, không lan man vào nhắc lại tiêu đề
        # snippet). Đã thử chỉ cắt n_predict trên adaptive.py gốc trước, không
        # ăn thua - cắt cụt thói quen echo tiêu đề giữa chừng thay vì loại bỏ
        # (xem README case "cà phê").
        step1_result = answer_entity_per_link(steps[0], **adaptive_kwargs)
        # adaptive.py đã tự phán "answer này có đáng tin không" qua `confident`
        # (đồng thuận đa link/đa nguồn) - dùng thẳng tín hiệu đó làm cổng chain,
        # không tự bịa thêm rule mới (vd check độ dài answer). Không confident
        # thì dừng, không chain tiếp với answer chưa chắc đúng - tránh lặp lại
        # bug đã thấy: answer thô dài dòng, agreement 0.12, vẫn bị nhét thẳng
        # vào {X} làm query bước 2 vỡ vụn.
        if not step1_result.get("confident"):
            if debug:
                print(
                    f"[debug][chain] bước 1 không confident (agreement="
                    f"{step1_result.get('agreement')}) - dừng chain, không sang bước 2",
                    flush=True,
                )
            return {"question": question, "type": "bridge", "steps": [
                {"query": steps[0], "result": step1_result},
            ]}
        step2_query = steps[1].replace("{X}", step1_result["answer"])
        # Câu hỏi khung ngoài cũng thường hỏi 1 tên riêng (thủ đô/tổng thống/
        # nhà vô địch...) - dùng lại adaptive_entity cho nhanh và dễ hội tụ
        # hơn, giống lý do đã sửa cho bước 1. Chỉ rớt về adaptive.py tổng quát
        # khi câu hỏi rõ ràng cần 1 con số ("bao nhiêu"/"how much"/"how many"),
        # vì prompt extract_entity ép trả lời 1 tên riêng - sai bản chất nếu
        # câu hỏi thật ra cần số liệu.
        step2_fn = answer_question_per_link if _expects_number(step2_query) else answer_entity_per_link
        if debug:
            print(
                f"[debug][chain] bước 1 confident (agreement={step1_result.get('agreement')}), "
                f"bước 2: {step2_query!r} ({step2_fn.__name__})",
                flush=True,
            )
        step2_result = step2_fn(step2_query, **adaptive_kwargs)
        return {"question": question, "type": "bridge", "steps": [
            {"query": steps[0], "result": step1_result},
            {"query": step2_query, "result": step2_result},
        ]}

    # comparison - các bước độc lập, không bước nào cần answer của bước khác.
    if debug:
        print(f"[debug][chain] type=comparison, {len(steps)} bước độc lập: {steps!r}", flush=True)
    return {"question": question, "type": "comparison", "steps": [
        {"query": s, "result": answer_question_per_link(s, **adaptive_kwargs)} for s in steps
    ]}


if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path

    import llm

    llm.MODEL_PATH = Path(__file__).parent / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

    # 3 câu hỏi khó đã dùng để tune adaptive.py (explore_adaptive.py) mà
    # decompose.py thực sự tách được (bridge/comparison) - các câu khó khác
    # trong bộ đó (giá vàng, Nobel, NATO, riêng tư, EN "difference between")
    # không có cấu trúc của/hơn/so-với nên decompose giữ nguyên, chạy như cũ,
    # không có gì để test riêng cho chain ở đây.
    QUESTIONS = [
        "Thủ đô của nước láng giềng phía bắc Việt Nam là gì?",
        "Tháp Eiffel cao hơn tháp Tokyo bao nhiêu mét?",
        "GDP bình quân đầu người Việt Nam năm 2025 so với năm 2015 tăng bao nhiêu lần?",
    ]
    if len(sys.argv) > 1:
        QUESTIONS = [" ".join(sys.argv[1:])]

    # max_n=50 (thay vì mặc định 8), consensus_threshold=0.80 (thay vì 0.75) -
    # đẩy câu hỏi khó tới giới hạn: search cho phép nhiều link hơn hẳn, đồng
    # thời khắt khe hơn khi nhận là confident. timeout_s tăng theo vì 50 link
    # (~17 batch N_PARALLEL=3) vượt xa timeout_s=90 mặc định.
    HARD_KWARGS = {"max_n": 50, "consensus_threshold": 0.80, "timeout_s": 900, "debug": True}

    llm.ensure_server()
    for q in QUESTIONS:
        t0 = time.monotonic()
        out = answer_chain(q, **HARD_KWARGS)
        elapsed = time.monotonic() - t0
        print(f"\n{'=' * 10} {q} (type={out['type']}, {elapsed:.1f}s) {'=' * 10}")
        for step in out["steps"]:
            r = step["result"]
            print(f"  [{step['query']}]")
            print(
                f"    -> {r.get('answer')!r} verdict={r.get('verdict')} "
                f"confident={r.get('confident')} agreement={r.get('agreement')} "
                f"bias={r.get('bias')} content_bias={r.get('content_bias')}"
            )
