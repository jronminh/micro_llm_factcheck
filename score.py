"""Heuristic rẻ để gắn nhãn tự động cho câu trả lời, thay việc tự đọc bằng mắt.

Không kiểm tra được câu trả lời có đúng thực tế hay không — chỉ bắt các
lỗi rõ ràng đã quan sát được qua nhiều lần test: lặp lại câu hỏi (echo),
copy nguyên tiêu đề snippet thay vì trả lời (title_echo), output vô nghĩa
kiểu "[2]" (degenerate), hoặc không bám vào nội dung đã search được
(ungrounded).
"""
import re
from difflib import SequenceMatcher

REFUSAL_MARKERS = (
    "không có thông tin", "khong co thong tin", "khong_co",
    "không được đáp ứng", "không đủ thông tin",
)
DEGENERATE_RE = re.compile(r"^\[?\d+\]?\.?$")


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def score_answer(question: str, answer: str, snippets: list[dict]) -> dict:
    answer = answer.strip()
    lower = answer.lower()

    if any(m in lower for m in REFUSAL_MARKERS):
        return {"verdict": "refusal", "echo_sim": None, "title_sim": None, "grounded_ratio": None}

    if not _words(answer) or DEGENERATE_RE.match(answer):
        return {"verdict": "degenerate", "echo_sim": None, "title_sim": None, "grounded_ratio": None}

    echo_sim = _similarity(question, answer)
    # Chỉ so với tiêu đề "dài kiểu bài báo" (>=4 từ) — tiêu đề ngắn trùng
    # answer thường là vì answer đúng chính là thực thể đó (VD "Paris"),
    # không phải model đang copy tiêu đề để trốn trả lời.
    long_titles = [s.get("title", "") for s in snippets if len(s.get("title", "").split()) >= 4]
    title_sim = max((_similarity(t, answer) for t in long_titles), default=0.0)

    a_words = _words(answer)
    snippet_words = set()
    for s in snippets:
        snippet_words |= _words(s.get("title", "") + " " + s.get("snippet", ""))
    grounded_ratio = len(a_words & snippet_words) / len(a_words) if a_words else 0.0

    # Tiêu đề tiếng Việt hay tự viết dạng câu hỏi (SEO), nên check title_echo
    # trước: một câu vừa giống tiêu đề vừa còn dấu "?" thì là copy tiêu đề,
    # không phải model đang lặp lại câu hỏi của người dùng.
    if title_sim >= 0.75:
        verdict = "title_echo"
    elif "?" in answer or echo_sim >= 0.92:
        # Câu trả lời thật hiếm khi còn giữ dấu "?" — mọi lần model lặp lại
        # câu hỏi quan sát được đều còn "?"; độ giống ký tự thô (echo_sim)
        # không đủ phân biệt vì câu trả lời hợp lệ cũng thường lặp lại chủ
        # ngữ của câu hỏi.
        verdict = "echo"
    elif grounded_ratio < 0.3:
        verdict = "ungrounded"
    else:
        verdict = "meaningful"

    return {
        "verdict": verdict,
        "echo_sim": round(echo_sim, 2),
        "title_sim": round(title_sim, 2),
        "grounded_ratio": round(grounded_ratio, 2),
    }
