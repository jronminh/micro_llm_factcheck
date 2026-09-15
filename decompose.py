"""Phân rã câu hỏi multi-hop bằng regex dựa trên từ nối (của/of, hơn/so với),
KHÔNG gọi model. Thay cho việc bắt LLM tự suy luận "khi nào cần tách" - đã thử
zero-shot và few-shot trong explore_decompose.py, cả hai đều không ổn định
(bỏ sót case cần tách, hoặc few-shot copy nhầm nội dung ví dụ thay vì câu hỏi
thật - xem case "phía Tây" lẽ ra phải là "phía Bắc"). Rule-based chỉ trích
substring trực tiếp từ câu hỏi thật, không sinh chữ mới -> không có kiểu lỗi
"tự tin nhưng sai nội dung" đó.

Bù lại, chỉ bắt được đúng những cấu trúc câu đã liệt kê dưới đây - câu hỏi
multi-hop không theo cấu trúc cố định (vd "Cầu thủ ghi bàn nhiều nhất WC 2026
chơi cho câu lạc bộ nào?" - không có "của"/"hơn") sẽ không bị tách, trả nguyên
câu gốc. Đây là lựa chọn có chủ đích: thà bỏ sót còn hơn tách sai.

Hai loại cấu trúc:

1. Genitive lồng nhau ("X của Y [là gì/là ai/ở đâu]" / "What is X of Y") - chỉ
   tách khi Y cần tra cứu riêng (vd "nước láng giềng phía bắc VN"), không tách
   khi Y đã là thực thể cụ thể sẵn (vd "Việt Nam") - tránh thêm 1 bước search
   thừa vô ích. Tín hiệu "cần tra cứu riêng" khác nhau giữa 2 ngôn ngữ (xem
   _needs_lookup_vn / _needs_lookup_en bên dưới - lý do tách hàm, không dùng
   chung 1 danh sách từ khóa).
2. So sánh 2 thực thể cùng thuộc tính ("A [tính từ] hơn B bao nhiêu [đơn vị]"
   / "A năm Y1 so với năm Y2 tăng/giảm bao nhiêu lần") - tách thành 2 truy vấn
   độc lập, mỗi truy vấn giữ nguyên thuộc tính/đại lượng đang hỏi.

Tín hiệu "cần tra cứu riêng" cho vế Y - khác nhau giữa 2 ngôn ngữ vì lý do ngữ
pháp, không phải chọn tùy tiện:

- Tiếng Việt là ngôn ngữ đơn lập/phân tích tính (analytic) - so sánh nhất luôn
  dùng 1 tiểu từ rời "nhất" gắn sau BẤT KỲ tính từ nào ("lớn nhất", "cao nhất",
  "nhiều nhất"...), không biến đổi hình thái tính từ. Nên chỉ cần 1 từ khóa
  "nhất" là bắt được mọi trường hợp so sánh nhất, không cần liệt kê từng tính
  từ. Giữ nguyên cách tiếp cận từ khóa (_RELATIVE_MARKERS), không cần NLP.
- Tiếng Anh là ngôn ngữ chắp dính/biến hình nhẹ (inflectional) ở tính từ so
  sánh nhất - hậu tố "-est" gắn dính vào từng tính từ khác nhau ("tallest",
  "largest", "richest"...), liệt kê từ khóa không bao giờ đủ. Cần POS tag thật
  (nltk) để bắt nhãn JJS/RBS (tính từ/trạng từ so sánh nhất) bất kể từ gốc là
  gì, cộng với nhãn WDT/WP/WRB (đại từ/trạng từ quan hệ - "that/which/who/
  where") cho mệnh đề quan hệ. Đây là lý do thêm dependency nltk chỉ cho nhánh
  tiếng Anh, không cần cho tiếng Việt.
"""
import re

import nltk

_RELATIVE_MARKERS_VN = (
    "mà", "giáp", "láng giềng", "tiếp giáp", "kế bên", "sát vách", "nhất",
)
_POS_LOOKUP_TAGS_EN = {"WDT", "WP", "WP$", "WRB", "JJS", "RBS"}

_GENITIVE_VN = re.compile(r"^(.+?)\s+của\s+(.+?)\s+(là gì|là ai|ở đâu|là bao nhiêu)\s*\??\s*$", re.IGNORECASE)
_GENITIVE_EN = re.compile(r"^(?:what|who)\s+is\s+(?:the\s+)?(.+?)\s+of\s+(?:the\s+)?(.+?)\s*\??\s*$", re.IGNORECASE)

_COMPARE_VN = re.compile(r"^(.+?)\s+(\S+)\s+hơn\s+(.+?)\s+(bao nhiêu(?:\s+[^\s?]+)?)\s*\??\s*$", re.IGNORECASE)
_COMPARE_TIME_VN = re.compile(
    r"^(.+?)\s+năm\s+(\d{4})\s+so với\s+năm\s+(\d{4})\s+(tăng|giảm)\s+bao nhiêu\s+lần\s*\??\s*$",
    re.IGNORECASE,
)


def _needs_lookup_vn(noun_phrase: str) -> bool:
    lower = noun_phrase.lower()
    return any(marker in lower for marker in _RELATIVE_MARKERS_VN)


def _needs_lookup_en(noun_phrase: str) -> bool:
    tags = {tag for _, tag in nltk.pos_tag(nltk.word_tokenize(noun_phrase))}
    return bool(tags & _POS_LOOKUP_TAGS_EN)


def decompose(question: str) -> list[str]:
    question = question.strip()

    m = _GENITIVE_VN.match(question)
    if m:
        outer, inner = m.group(1).strip(), m.group(2).strip()
        if _needs_lookup_vn(inner):
            return [inner, f"{outer} của {{X}} {m.group(3)}".strip()]

    m = _GENITIVE_EN.match(question)
    if m:
        outer, inner = m.group(1).strip(), m.group(2).strip()
        if _needs_lookup_en(inner):
            return [inner, f"{outer} of {{X}}?".strip()]

    m = _COMPARE_VN.match(question)
    if m:
        entity_a, adjective, entity_b, quantity = (g.strip() for g in m.groups())
        return [f"{entity_a} {adjective} {quantity}?", f"{entity_b} {adjective} {quantity}?"]

    m = _COMPARE_TIME_VN.match(question)
    if m:
        metric, year_a, year_b, _direction = (g.strip() for g in m.groups())
        return [f"{metric} năm {year_a} là bao nhiêu?", f"{metric} năm {year_b} là bao nhiêu?"]

    return [question]


EXTRA_EN_QUESTIONS = [
    ("EN so sánh nhất (-est, test JJS)", "What is the population of the largest city in Japan?"),
    ("EN so sánh nhất khác từ gốc (test JJS tổng quát hóa)", "What is the GDP of the richest country in Europe?"),
    ("EN control, entity đã cụ thể (KHÔNG nên tách)", "What is the capital of France?"),
]

if __name__ == "__main__":
    from explore_decompose import QUESTIONS

    for kind, q in QUESTIONS + EXTRA_EN_QUESTIONS:
        steps = decompose(q)
        tag = "TÁCH" if len(steps) > 1 else "giữ nguyên"
        print(f"[{kind}] {q}")
        print(f"  -> ({tag}) {steps}")
        print()
