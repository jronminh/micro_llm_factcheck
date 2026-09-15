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

## Kết quả benchmark (Samsung SM-S7110, 7 core, n=8 câu hỏi)

`llama-bench` baseline (không search, không qua HTTP): pp512 ~24 t/s,
tg128 ~6.8 t/s. Sau đó chạy full pipeline (search + server) cho 8 câu hỏi
đa dạng lĩnh vực, server giữ warm giữa các lần:

| câu hỏi | search_s | model_s | gen_tps | câu trả lời |
|---|---|---|---|---|
| Dân số VN 2026 | 1.12 | 28.3 | 8.7 | "102.457.889 người, theo đoạn trích [1]" |
| Tổng thống Mỹ hiện tại | 0.86 | 18.8 | 8.9 | "Donald Trump" |
| Giá vàng hôm nay | 0.86 | 53.2 | 0.5 | lặp lại tiêu đề snippet (hỏng) |
| Chiều cao Everest | 1.80 | 24.4 | 7.0 | "8.848,86 mét" |
| Tỷ giá USD/VND | 1.08 | 24.6 | 6.3 | "25.750đ" |
| World Cup 2026 | 0.93 | 24.3 | 6.3 | "Mỹ, Canada và Mexico" |
| Python là ngôn ngữ gì | 1.05 | 32.2 | 7.0 | tóm tắt đúng |
| Nhiệt độ Hà Nội hiện tại | 1.06 | 19.1 | 8.0 | "không có thông tin cụ thể... không được đáp ứng" |

Thống kê: `search_s` avg 1.09 (0.86-1.80), luôn nhanh và ổn định. `model_s`
avg 28.1 (18.8-53.2). `gen_tps` avg 6.60, median 7.00, nhưng có 1 outlier
rơi xuống 0.5 — trùng đúng lúc RAM pressure cao nhất (máy lúc đo chỉ
~1.6GB available/7.1GB, swap 5.2/11GB) và cũng là câu trả lời hỏng nhất.

Hai quan sát đáng chú ý:

1. **Ràng buộc "chỉ dùng snippet" hoạt động đúng thiết kế** — câu hỏi về
   nhiệt độ Hà Nội, model tự nhận không đủ thông tin và từ chối trả lời
   thay vì bịa, đúng ý ban đầu của ý tưởng grounding.
2. **RAM pressure có thể ảnh hưởng cả chất lượng, không chỉ tốc độ** — outlier
   0.5 t/s trùng với câu trả lời tệ nhất (lặp tiêu đề snippet). Mới quan sát
   được 1 lần, n=8 chưa đủ để khẳng định tương quan, cần thêm dữ liệu.

## So sánh 1.5B vs 0.5B

`compare_models.py` chạy cùng 3 câu hỏi lần lượt trên mỗi model (dừng
server, đổi model, khởi động lại, để không chạy đồng thời 2 model):

| model | total_s (avg) | gen_tps (avg) |
|---|---|---|
| 1.5B | 30.1 | 7.01 |
| 0.5B | 10.0 | 17.12 |

0.5B nhanh hơn ~2.44x về gen_tps và giảm total_s xuống 1/3. Câu trả lời của
0.5B với 3 câu hỏi test (Everest, tổng thống Mỹ, Python) vẫn đúng và bám
sát snippet, không thấy khác biệt rõ về chất lượng ở mức test nhỏ này —
cần thử nhiều câu hỏi khó hơn (đa bước, số liệu) để thấy giới hạn thật của
0.5B.

## Tiếng Anh vs tiếng Việt (0.5B)

So sánh 4 cặp câu hỏi cùng nội dung, một bằng tiếng Việt một bằng tiếng
Anh (`Dân số VN 2026`, `Everest`, `World Cup 2026`, `thủ đô Pháp`). 3/4
cặp cho chất lượng tương đương ở cả hai ngôn ngữ; chỉ 1 cặp (dân số VN)
tiếng Việt bị hỏng (model lặp lại câu hỏi, không ra số liệu) trong khi
tiếng Anh trả lời đúng. `gen_tps` giữa hai ngôn ngữ không khác biệt đáng
kể. Với n=4, chưa đủ để khẳng định tiếng Anh tốt hơn hệ thống — có thể chỉ
là 0.5B ngẫu nhiên yếu ở câu có số liệu phức tạp, không phải do ngôn ngữ.
Cần thêm dữ liệu nếu muốn kết luận chắc.

## Test nhiều dạng câu hỏi + mode "extractive" (0.5B)

`explore_formats.py` chạy 8 dạng câu hỏi khác nhau (câu hỏi đầy đủ, có/không,
kiểu từ khóa như gõ vào ô search, so sánh, liệt kê, mở/rộng, ngoài phạm vi,
định nghĩa ngắn) qua 2 mode prompt:

- **synth**: model được phép ghép câu, miễn bám sát snippet (như hiện tại).
- **extract**: ép model chỉ được copy nguyên văn một đoạn trong snippet,
  không diễn giải, không ghép câu — cố tình cho nó hoạt động gần với một
  "công cụ tìm kiếm" hơn là một model đang suy luận, đúng ý tưởng gốc "chỉ
  được search, không được suy luận".

Kết quả: **synth 6/8 đúng, extract chỉ 3/8 đúng rõ ràng.** Ép 0.5B vào vai
"chỉ trích xuất nguyên văn" làm giảm độ tin cậy, không tăng — lỗi thường
gặp ở mode extract: model trả về `[2]` (chỉ số thứ tự snippet, không có
nội dung), hoặc trích tiêu đề snippet (trùng cấu trúc với câu hỏi) thay vì
nội dung trả lời thật. Ở cả 2 mode, khi search không có snippet đủ tốt,
model có xu hướng lặp lại câu hỏi thay vì tuân theo chỉ dẫn "nói rõ không
có thông tin" / in `KHONG_CO` — cho thấy 0.5B theo instruction có điều
kiện ("nếu... thì...") kém ổn định, đây là giới hạn thật của quy mô model,
không phải vấn đề prompt engineering.

Điểm tích cực: câu hỏi kiểu từ khóa ngắn (không phải câu hoàn chỉnh, ví dụ
"dân số hà nội 2026") hoạt động tốt ở cả 2 mode — đúng với việc dùng model
gần với một search engine thật.

## Tune adaptive.py với câu hỏi khó (0.5B)

`explore_adaptive.py` chạy 8 câu hỏi khó hơn mức đã test trước đó qua
`answer_question_per_link`: multi-hop (cần suy luận địa lý 2 bước), số liệu
biến động (giá vàng), so sánh cần tính toán, tin gần đây, phủ định + kiến
thức cần cập nhật, câu nên bị từ chối (riêng tư), tính toán nhiều mệnh đề
theo thời gian, và một câu tiếng Anh phức tạp. Chạy trên máy này cần
Termux ở foreground (xem lưu ý CPU throttling ở trên) - lúc đó tốc độ ổn
định 3-46s/câu, không còn timeout 20s.

Kết quả ban đầu (`consensus_threshold=0.6`, `min_votes=3`): 3/8 câu
**"confident" nhưng sai** - nguy hiểm nhất với một tool tự nhận là
"fact-check":

- "Thủ đô nước láng giềng phía bắc VN?" -> "Lào" (đúng là Trung Quốc; Lào ở
  phía Tây/Tây Nam) - agreement 0.67.
- "Anh/Pháp/Thụy Điển, nước nào không thuộc NATO?" -> "Anh" (thực ra sau
  2024 cả 3 đều thuộc NATO - câu hỏi test có premise lỗi thời, nhưng model
  cũng không phát hiện ra) - agreement 0.67.
- "GDP đầu người 2025 so với 2015 tăng mấy lần?" -> chia sai, lấy USD chia
  cho **năm** (2015) thay vì GDP năm 2015 - agreement 0.6, đúng ngay ngưỡng.

Tăng `consensus_threshold` lên 0.75 và `min_votes` lên 4, chạy lại: case
GDP (lỗi tính toán ngẫu nhiên giữa các lần sinh) biến mất - đúng như dự
đoán, ngưỡng chặt hơn triệt tiêu được sự đồng thuận ngẫu nhiên. Nhưng 2
case địa lý/NATO **vẫn confident=True và vẫn sai** - agreement chạm đúng
0.75 vì nhiều nguồn/lần sinh cùng lặp lại một lỗi giống nhau một cách có hệ
thống (echo chamber giữa các trang SEO tiếng Việt copy nhau, hoặc model
liên tục nhầm cùng một kiểu địa lý).

**Kết luận quan trọng**: agreement/consensus voting chỉ triệt tiêu được sai
số *ngẫu nhiên, độc lập giữa các lần sinh* (như lỗi tính toán GDP) - không
triệt tiêu được sai số *có hệ thống, tương quan giữa các nguồn* (nhiều
nguồn cùng sai theo cùng một cách). Đây là giới hạn toán học của voting
theo đa số (majority voting không khử được correlated error), không phải
bug tham số - tăng threshold/min_votes cao hơn nữa cũng không chắc giải
quyết được nếu lỗi lặp lại ở >75% nguồn. Câu hỏi tốt tin: các case
"không đồng thuận" (giá vàng, so sánh Eiffel/Tokyo, Nobel 2026 - model bịa
"Nguyễn Thị Minh Khai", câu riêng tư, câu tiếng Anh) đều đúng bị đánh
`confident=False`, đúng ý thiết kế.

Đã cập nhật default trong `adaptive.py`: `consensus_threshold=0.75`,
`min_votes=4` (giữ - vẫn có ích cho lỗi ngẫu nhiên dù không phải toàn bộ
giải pháp).

### Đẩy đến giới hạn: mở rộng max_n/min_votes/threshold có cứu được 2 case còn sai không?

Thêm xử lý song song theo batch (`N_PARALLEL=3` link/lúc, `llama-server -np 3`,
`-c 3072`) để chạy được `max_n` lớn trong thời gian hợp lý - phát hiện kèm 1
race condition thật: `ensure_server()` không thread-safe, nhiều thread cùng
gọi lúc server chưa sẵn sàng sẽ cùng tự Popen process mới, giành port. Đã
khóa bằng `threading.Lock`.

Chạy lại 3 câu hỏi còn sai với điều kiện khắc khe hơn nhiều:
`max_n=15` (search trả 10), `min_votes=8`, `consensus_threshold=0.85`:

- **Multi-hop** ("thủ đô nước láng giềng phía bắc VN?"): đi qua 9 link,
  agreement **tăng lên 0.89** (v.11 test ở n nhỏ chỉ 0.67) - "Lào" vẫn sai,
  nhưng tự tin hơn. Soi nội dung: 5/9 nguồn nói "Lào", 3/9 nói "Hà Nội"
  (cũng sai), chỉ 1/9 nói đúng "Trung Quốc". Nhiều trang giáo án/SEO tiếng
  Việt cùng lặp lại một lỗi giống nhau (có vẻ từ cùng dạng bài địa lý lớp 5)
  - mở rộng search không cứu được vì lỗi nằm ở corpus bị lệch, không phải
  ở model hay ở tham số voting.
- **NATO**: agreement giảm xuống 0.40 (đúng, hết confident) - nguồn chia
  phe ~50/50 giữa "Anh" và "Thụy Điển" (khác nhau do bài viết cũ/mới, lỗi
  độc lập giữa nguồn, không phải cùng một lỗi lặp lại).
- **GDP**: vẫn confident=False (0.40) như ở lần tune trước, thêm lộ ra 2/10
  nguồn cho đáp án khác hẳn ("tăng 7 lần") - tín hiệu bổ sung chưa khai thác.

**Kết luận quan trọng nhất**: hệ thống không phân biệt được "đồng thuận vì
đúng" và "đồng thuận vì nhiều nguồn cùng lặp lại một lỗi giống nhau". Mở
rộng max_n/min_votes/threshold chỉ giúp khi lỗi ngẫu nhiên/độc lập giữa
nguồn (case NATO, GDP) - khi lỗi mang tính hệ thống ở cấp corpus search
(case multi-hop), mở rộng tìm kiếm có thể khiến kết luận sai tự tin hơn,
phản trực giác. Đây là giới hạn kiến trúc thật của "search + voting" với
model nhỏ, không phải bug tham số - cần một cơ chế khác (ví dụ: đối chiếu
với nguồn có uy tín cao hơn, hoặc phát hiện câu hỏi dạng multi-hop để hạ
tin cậy mặc định) nếu muốn giải quyết, không chỉ chỉnh số.

## Trạng thái

Bản chạy được đầu tiên hoàn chỉnh: search + model (server warm) + benchmark
ghi log. Đã test: 8 câu hỏi đa dạng lĩnh vực với 1.5B; so sánh tốc độ 1.5B
vs 0.5B; tiếng Anh vs tiếng Việt; 8 dạng câu hỏi khác nhau ở 2 mode prompt
(synth/extract) với 0.5B; tune `adaptive.py` (per-link consensus) với 8 câu
hỏi khó hơn.

Phát hiện quan trọng nhất: **ép model nhỏ chỉ được "search, không được suy
luận" theo nghĩa chặt (mode extract) làm giảm độ tin cậy so với cho nó
tổng hợp nhẹ (mode synth)** — 0.5B không đủ khả năng theo instruction có
điều kiện phức tạp một cách ổn định. Đây ngược với giả định ban đầu của ý
tưởng, và là giới hạn thật của quy mô model, không phải vấn đề prompt.

Phát hiện thứ hai (từ tune adaptive.py): **consensus voting giữa nhiều
nguồn/lần sinh chỉ khử được lỗi ngẫu nhiên, không khử được lỗi hệ thống
lặp lại giống nhau giữa các nguồn** - "confident" không đồng nghĩa "đúng"
ở các câu multi-hop hoặc dùng kiến thức có thể lỗi thời.

Bước tiếp theo hợp lý: thử mode extract với model 1.5B (có tuân theo
instruction có điều kiện tốt hơn không?); chạy nhiều câu hỏi hơn để xác
nhận tương quan RAM pressure/chất lượng câu trả lời; thử câu hỏi tiếng
Anh/Việt với n lớn hơn để kết luận chắc về sự khác biệt ngôn ngữ; thử lại
2 case multi-hop/NATO còn sai với model 1.5B để xem lỗi có phải do quy mô
model hay do search/snippet chất lượng thấp.
