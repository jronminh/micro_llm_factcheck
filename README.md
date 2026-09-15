# Micro LLM Fact-Check

Ý tưởng: dùng một model LLM rất nhỏ (dưới 3B tham số), chạy local, ghép với
một search engine. Python đóng vai trò điều phối và giới hạn model chỉ được
search rồi tổng hợp kết quả, không được tự trả lời từ kiến thức nền hoặc suy
luận ngoài nội dung search được.

## Vấn đề đang giải quyết

Model nhỏ hay bị hallucination khi trả lời trực tiếp từ tri thức nội tại vì
dung lượng tham số thấp. Nếu bắt nó chỉ tổng hợp trên nội dung đã search
được (grounding), độ chính xác tăng lên đáng kể so với hỏi trực tiếp.

## Giới hạn cần hiểu rõ

Python không thể ép model "không được sai" ở tầng sinh token — cơ chế sinh
text không có khái niệm đúng/sai. Hai việc Python thực sự làm được:

1. **Giới hạn đầu vào** — model chỉ nhìn thấy đoạn text đã search được,
   không có gì khác trong context.
2. **Kiểm tra đầu ra sau khi sinh** — bắt model trích dẫn nguyên văn từ
   nguồn thay vì diễn giải tự do, rồi đối chiếu câu trả lời với nguồn gốc để
   loại bỏ phần không có căn cứ.

Ràng buộc "chỉ được search, không được trả lời/suy luận" chỉ là ràng buộc
mềm qua prompt và luồng điều khiển, model vẫn có thể lách nếu search không
ra kết quả tốt.

## Hướng kỹ thuật

- **Model**: chạy qua `llama.cpp` hoặc Ollama, bản quantize (Q4). Vùng an
  toàn cho chất lượng tổng hợp: 1.5B-3B tham số. Dưới 1B thường quá yếu để
  tổng hợp mạch lạc.
- **Search**: SearXNG tự host hoặc DuckDuckGo, không cần API key.
- **Điều phối**: script Python — nhận câu hỏi, gọi search, lọc/rank
  snippet, đưa vào prompt ép trích dẫn, gọi model, validate output so với
  nguồn.

## Ràng buộc môi trường

Chạy trên Termux/Android (điện thoại). Model từ 3B trở lên trên CPU sẽ chậm
(vài chục giây mỗi câu trả lời). Cần cân nhắc giữa chất lượng và thời gian
phản hồi khi chọn kích thước model.

## Dự án tương tự

Ý tưởng grounding LLM bằng search không mới, đã có nhiều dự án mã nguồn mở:

- [Perplexica](https://github.com/topics/searxng?o=desc&s=stars) — clone mã
  nguồn mở của Perplexity, 20k+ stars, dùng SearXNG + LLM để search và tổng
  hợp có trích dẫn nguồn. Gần nhất với ý tưởng này về kiến trúc.
- [Farfalle](https://github.com/rashadphz/farfalle) — self-host AI search
  engine, hỗ trợ local LLM (Llama 3, Gemma, Mistral, Phi-3) hoặc cloud.
- [GroundedLLM](https://github.com/wsargent/groundedllm) — agent được
  "ground" bằng search + extract tool để giảm hallucination, gần với mục
  tiêu "chỉ được search, không được tự suy luận".
- [Sova](https://github.com/LexiestLeszek/sova_ollama) — RAG-based web
  search engine dùng Ollama + scraping Google, quy mô nhỏ gọn.
- [RAGFlow](https://github.com/infiniflow/ragflow) — RAG engine lớn hơn,
  thiên về document search có agent, không tập trung vào web search.

Điểm khác biệt (gap) trong ý tưởng này: các dự án trên đều nhắm tới model
tầm trung/lớn (Llama 3, GPT-4, Mistral) để có chất lượng tổng hợp tốt.
Không có dự án nào tập trung cụ thể vào model rất nhỏ (dưới 3B) kết hợp
ràng buộc kiến trúc cứng "chỉ search, không suy luận" — hầu hết chỉ ràng
buộc qua prompt.

## Trạng thái

Mới ở giai đoạn ý tưởng, chưa có code.
