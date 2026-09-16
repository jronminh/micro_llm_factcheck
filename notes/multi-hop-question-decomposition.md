# Phân rã câu hỏi multi-hop: vì sao dùng regex+POS thay vì LLM

Tổng hợp lại khảo sát trong `explore_decompose.py`/`decompose.py`
(commit `e990670`) - 4 cách đã thử, 3 bị loại, chốt ở cách thứ 4. Đáng nhớ
vì đi ngược trực giác "cứ dùng LLM cho linh hoạt": ở quy mô model 0.5B,
rule-based lại đáng tin hơn hẳn.

## 4 cách đã thử

1. **LLM tự sinh câu hỏi con** (zero-shot, prompt yêu cầu xuất JSON
   `{"steps": [...]}`) - model **hallucinate nội dung**: sinh ra câu hỏi con
   không tồn tại trong câu gốc, hoặc lặp lại y nguyên câu gốc 2 lần khi đáng
   lẽ phải tách theo mốc thời gian/thực thể khác nhau (case GDP 2025 vs
   2015 - xem log `explore_decompose.py`).
2. **LLM tự sinh, có few-shot examples** - đỡ hallucinate nội dung hoàn
   toàn mới, nhưng phát sinh lỗi khác: **copy nhầm nội dung ví dụ** thay vì
   xử lý câu hỏi thật. Case cụ thể đã bắt được: câu hỏi tiếng Anh "...borders
   Vietnam to the **north**" bị model tách thành "nước nào giáp Việt Nam ở
   phía **Tây**" - rõ ràng lẫn với 1 ví dụ few-shot có hướng khác trong
   prompt, không phải suy luận từ câu hỏi thật.
3. **LLM chỉ gắn nhãn** (không tự sinh câu hỏi con, chỉ đánh dấu cụm từ có
   sẵn trong câu gốc cần tách) - giảm được rủi ro hallucinate nội dung mới,
   nhưng vẫn ~40% ra nhãn rác (label không map được về cụm từ thật trong câu
   gốc).
4. **Rule-based: regex cho tiếng Việt + POS tag thật (nltk) cho tiếng Anh**
   - không gọi model, chỉ trích xuất substring trực tiếp từ câu hỏi thật nên
   *không có kiểu lỗi "tự tin nhưng sai nội dung"* của 3 cách trên. Đánh đổi:
   chỉ bắt được đúng cấu trúc câu đã liệt kê trước, câu multi-hop không theo
   khuôn cố định sẽ bị bỏ sót (trả nguyên câu gốc) thay vì tách sai - lựa
   chọn có chủ đích ("thà bỏ sót còn hơn tách sai"). Test: 10/10 case đúng,
   không tách sai case nào.

## Vì sao tiếng Việt và tiếng Anh cần 2 cơ chế nhận diện khác nhau

Không phải chọn tuỳ tiện - do khác biệt loại hình ngôn ngữ (typology) thật:

- **Tiếng Việt** là ngôn ngữ đơn lập/phân tích tính (analytic) - so sánh
  nhất luôn dùng 1 tiểu từ rời "nhất" gắn sau BẤT KỲ tính từ nào ("lớn
  nhất", "cao nhất", "nhiều nhất"...), tính từ không biến đổi hình thái. Chỉ
  cần 1 từ khóa "nhất" là bắt được mọi trường hợp so sánh nhất - không cần
  liệt kê từng tính từ, không cần NLP thật.
- **Tiếng Anh** là ngôn ngữ chắp dính/biến hình nhẹ (inflectional) ở tính từ
  so sánh nhất - hậu tố "-est" gắn dính vào từng tính từ khác nhau
  ("tallest", "largest", "richest"...) - liệt kê từ khóa không bao giờ đủ
  (không thể liệt kê hết mọi tính từ tiếng Anh). Cần POS tag thật (`nltk`)
  để bắt nhãn `JJS`/`RBS` (tính từ/trạng từ so sánh nhất) bất kể từ gốc là
  gì, cộng nhãn `WDT`/`WP`/`WRB` (đại từ/trạng từ quan hệ - "that/which/
  who/where") cho mệnh đề quan hệ kiểu "the country that borders...".

Đây là lý do `decompose.py` có 2 hàm `_needs_lookup_vn`/`_needs_lookup_en`
riêng biệt thay vì dùng chung 1 danh sách từ khóa - không phải trùng lặp
code thừa, mà do 2 cơ chế ngữ pháp thật sự khác nhau.

## Bài học tổng quát

Khi model nhỏ (0.5B) cần thực hiện 1 tác vụ có cấu trúc rõ ràng, hữu hạn
(ở đây: nhận diện + tách câu theo vài khuôn cú pháp cố định), rule-based
based trên đặc điểm ngôn ngữ học thật của câu hỏi input đáng tin hơn hẳn so
với để model tự suy luận - kể cả khi có few-shot. Model nhỏ không đủ khả
năng phân biệt "khi nào nội dung ví dụ kết thúc, câu hỏi thật bắt đầu" một
cách ổn định. Liên quan tới phát hiện chung của dự án (xem README): model
0.5B không theo tốt instruction có điều kiện phức tạp.
