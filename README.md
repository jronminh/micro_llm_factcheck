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

## Cách chạy

```
bash scripts/setup.sh              # cài llama-cpp, pip deps, tải model (idempotent, có checksum)
llama-bench -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf   # baseline tok/s thô, không qua search
python search.py "câu hỏi"         # test riêng phần search
python pipeline.py "câu hỏi"       # search + model, in câu trả lời
python benchmark.py "câu hỏi"      # như trên, kèm đo thời gian, ghi vào benchmark/results.csv
```

Model: `Qwen/Qwen2.5-1.5B-Instruct-GGUF`, bản `q4_k_m` (~1.1GB). Chạy qua
`llama-server` (HTTP, OpenAI-compatible, CPU only) — `pipeline.py` tự khởi
động server nếu chưa chạy và giữ nó chạy warm giữa các lần gọi tiếp theo.
Search: DuckDuckGo HTML endpoint, không cần API key.

Lý do không gọi `llama-cli` trực tiếp: ở chat mode, `llama-cli` echo lại
prompt vào stdout xen với câu trả lời (và cắt ngắn phần echo khi prompt
dài), không có ranh giới rõ ràng để tách câu trả lời bằng regex.
`llama-server` trả JSON sạch (`choices[0].message.content` + `timings`).

## Kết quả benchmark ban đầu (Samsung SM-S7110, 7 core)

`llama-bench` baseline (không search, không qua HTTP): pp512 ~24 t/s,
tg128 ~6.8 t/s. Sau đó chạy full pipeline (search + server) cho 3 câu hỏi,
server giữ warm giữa các lần:

| câu hỏi | search_s | model_s | gen_tps |
|---|---|---|---|
| Dân số VN 2026 | 1.12 | 28.3 | 8.7 |
| Tổng thống Mỹ hiện tại | 0.86 | 18.8 | 8.9 |
| Giá vàng hôm nay | 0.86 | 53.2 | 0.5 |

Search luôn nhanh (dưới 1.2s). `gen_tps` dao động rất mạnh cho cùng một
model đã warm (8.9 t/s xuống 0.5 t/s) — RAM pressure trên máy (lúc đo chỉ
~1.6GB available/7.1GB, swap 5.2/11GB) ảnh hưởng performance nhiều hơn cả
kích thước model. Câu hỏi "giá vàng" cũng ra câu trả lời hỏng (lặp lại tiêu
đề snippet thay vì tổng hợp) đúng lúc gen_tps thấp nhất — nghi ngờ có liên
quan, cần thêm dữ liệu để xác nhận.

## Trạng thái

Bản chạy được đầu tiên hoàn chỉnh: search + model (server warm) + benchmark
ghi log. Model 1.5B chạy được trên Termux, prompt grounding hoạt động đúng
(model trích dẫn "[1]" khi có snippet phù hợp). Bước tiếp theo hợp lý: chạy
nhiều câu hỏi hơn để xác nhận tương quan giữa RAM pressure và chất lượng
câu trả lời, và/hoặc thử model 0.5B để so sánh.
