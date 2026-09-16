# Micro LLM Fact-Check

Ý tưởng: dùng một model LLM rất nhỏ (dưới 3B tham số), chạy local, ghép với
một search engine. Python đóng vai trò điều phối và giới hạn model chỉ được
search rồi tổng hợp kết quả, không được tự trả lời từ kiến thức nền hoặc suy
luận ngoài nội dung search được.

File này là nhật ký tiến độ dự án (theo commit). Nghiên cứu nền/tham khảo
sâu hơn về từng chủ đề kỹ thuật cụ thể (không phải kết quả thực nghiệm của
dự án) nằm ở `notes/` - xem `notes/android-thermal-apis.md` (2 lớp API
nhiệt Android, ý nghĩa từng loại cảm biến), `notes/android-background-app-throttling.md`
(Android hạ xung app chạy nền, dễ tưởng nhầm RAM/nhiệt),
`notes/llamacpp-hardware-acceleration-android.md` (CPU dotprod/i8mm, GPU
OpenCL, NPU Hexagon - cái nào đáng làm), `notes/multi-hop-question-decomposition.md`
(vì sao chọn regex+POS thay vì LLM để tách câu hỏi),
`notes/termux-pgrep-pkill-exact-match-quirk.md` (vì sao `pkill -x` không
dùng được trên máy này), `notes/micro-llm-alternatives-to-qwen.md`
(khảo sát model nhỏ khác ngoài Qwen2.5, tháng 9/2026), và
`notes/qwen3-reasoning-chain-adaptive.md` (thí nghiệm chain 3 bước tận dụng
reasoning của Qwen3 - chưa hoàn thiện, xem trạng thái trong note).

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

### Thêm 2 chỉ số thống kê: coverage và bias

`agreement` (cỡ cụm thắng / số link *có phản hồi*) không lộ ra khi nhiều
link bị timeout/lỗi, và không phân biệt được đồng thuận độc lập thật với
đồng thuận do trùng nguồn. Thêm 2 chỉ số vào output của
`answer_question_per_link`, tính hoàn toàn từ dữ liệu đã có, không tốn thêm
lượt gọi model:

- **coverage** = cỡ cụm thắng / tổng số link **đã thử** (kể cả timeout/lỗi,
  qua counter `n_attempted`) - khác `agreement` khi có link fail; agreement
  cao mà coverage thấp nghĩa là nhiều link đã "rớt" trước khi tính đồng thuận.
- **bias** = 1 - (số domain khác nhau / cỡ cụm thắng) trong cụm thắng, parse
  hostname từ `url` của snippet (`urllib.parse`). 0 = mọi link trong cụm đến
  từ domain khác nhau (đồng thuận độc lập), gần 1 = cụm thắng chủ yếu dựa
  vào rất ít domain lặp lại (rủi ro echo-chamber).

Test lại case NATO: `bias=0.0` - mọi domain trong cụm thắng khác nhau (2
phiên bản Wikipedia, ai-hay.vn, tuoitre.vn...) - khớp đúng, đây là bất đồng
thật giữa nguồn độc lập.

Test lại case multi-hop (chạy lại, search trả bộ link khác do DDG không ổn
định giữa các lần gọi - lần này đa số ra đúng "Bắc Kinh"): `bias=0.12`, khá
thấp, dù đã biết nội dung các nguồn tương quan cao (đều paraphrase cùng một
bài "Địa Lí lớp 5 - Bài 19"). **Hạn chế thật đã lộ ra**: bias theo domain chỉ
bắt được kiểu tương quan "cùng 1 site lặp lại", không bắt được kiểu sâu hơn
"nhiều domain độc lập nhưng cùng chép/paraphrase một nguồn gốc" (ví dụ cùng
một sách giáo khoa) - đây chính là dạng tương quan đã gây ra case "Lào" sai
mà agreement cao ở lần test trước. Muốn bắt được dạng này cần so khớp nội
dung snippet gốc (không chỉ domain), chưa làm.

### Thêm content_bias: so khớp nội dung snippet gốc, không chỉ domain

Tái dùng `_cluster()` (đang dùng để gom câu trả lời giống nhau) cho
title+snippet gốc của các nguồn trong cụm thắng - `content_bias` = tỉ lệ
cụm nội dung lớn nhất / cỡ cụm thắng, cùng thang đo với `bias` (0 = mọi
snippet khác nhau thật, gần 1 = phần lớn là bản chép/diễn giải của nhau).

Test lại case multi-hop: `bias` (domain) = 0.0 (9 domain khác nhau), nhưng
`content_bias` = **0.5** - bắt được đúng cái domain-bias bỏ lỡ: nhìn snippet
thật, phần lớn đều mở đầu giống nhau kiểu "Bài 19: Các nước láng giềng của
Việt Nam..." / "Giáo án Địa lí lớp 5...", tức cùng một giáo án gốc dù đăng
trên domain khác nhau. Case NATO (nguồn thật độc lập): `content_bias` = 0.25,
thấp hơn rõ ràng - phân biệt được 2 case tốt.

Lưu ý quan trọng: content_bias **không suy ra đúng/sai** - test case dễ
"Thủ đô Việt Nam là gì?" (câu trả lời đúng) vẫn ra `content_bias=0.4` vì
nhiều nguồn diễn đạt giống nhau khi cùng nói một fact đơn giản, phổ biến.
Đây chỉ là chỉ số phụ trợ đo độ độc lập của bằng chứng, không thay được
việc phải nhìn cả agreement + bias + content_bias cùng nhau.

## Phân rã câu hỏi multi-hop (decompose.py)

Hướng đầu vào: thay vì search 1 lần cho cả câu hỏi rồi trông chờ voting/bias
sửa lỗi hệ thống (không sửa được - xem case "thủ đô nước láng giềng phía bắc
VN" ở trên), thử tách câu hỏi multi-hop thành các câu hỏi con để search riêng
từng bước *trước khi* search. Thử 4 cách tiếp cận, từ để model tự sinh đến
thuần quy tắc:

1. **LLM tự sinh câu hỏi con (mode `decompose`, zero-shot)**: đúng format JSON
   100%, nhưng bỏ sót chính case multi-hop cần tách nhất, sinh placeholder
   degenerate ("câu hỏi con 1"), hoặc mất thuộc tính khi tách câu so sánh (mất
   "chiều cao" khi tách "Eiffel cao hơn Tokyo Skytree bao nhiêu mét").
2. **LLM tự sinh, few-shot (3 ví dụ)**: sửa được 2 lỗi trên, nhưng lộ lỗi nguy
   hiểm hơn - câu hỏi multi-hop tiếng Anh ("...borders Vietnam **to the
   north**") bị model **copy nhầm nội dung ví dụ** ("phía Tây" từ ví dụ, thay
   vì "phía Bắc" từ câu hỏi thật) - lỗi tự tin nhưng sai nội dung, nguy hiểm
   hơn hẳn lỗi degenerate rõ ràng vì không lọc được bằng `score.py`.
3. **LLM chỉ gắn nhãn cụm có sẵn (mode `tag`), Python tự ráp câu hỏi con**:
   giảm việc "sinh" xuống còn "định vị" (gần giống mode extract). Tốt hơn về
   lý thuyết nhưng thực tế vẫn ~4/10 case ra tag rác hoặc JSON key trùng lặp,
   một case còn tự dịch cả câu hỏi tiếng Anh sang tiếng Việt rồi tag bản dịch.
   Điểm sáng: bắt đúng được case genitive ẩn (không có "của" tường minh) mà
   regex thuần không làm được.
4. **Quy tắc thuần (regex + từ loại thật cho tiếng Anh), không gọi model**:
   `decompose.py`. Bắt 2 loại cấu trúc: (a) genitive lồng nhau "X của
   Y [là gì/là ai/ở đâu/là bao nhiêu]" / "What is X of Y" - chỉ tách khi Y
   cần tra cứu riêng; (b) so sánh 2 thực thể "A hơn B bao nhiêu" / so sánh
   theo 2 mốc thời gian "...năm A so với năm B tăng/giảm bao nhiêu lần".
   Tín hiệu "Y cần tra cứu riêng" **khác nhau có chủ đích giữa 2 ngôn ngữ**:
   tiếng Việt phân tích tính, so sánh nhất luôn dùng tiểu từ rời "nhất" nên
   1 từ khóa bắt hết mọi tính từ; tiếng Anh biến hình "-est" theo từng từ
   ("tallest"/"largest"/"richest"...) nên từ khóa không đủ, phải dùng POS tag
   thật (`nltk`, nhãn JJS/RBS/WDT/WP) mới tổng quát hóa được. Kết quả: 10/10
   case test (VN + EN, gồm cả 2 case so sánh nhất dùng từ gốc khác nhau) đúng,
   không case nào tách sai - đổi lại bỏ sót có chủ đích các case không theo
   cấu trúc đã liệt kê (vd nối bằng động từ như "...chơi cho câu lạc bộ nào?",
   hoặc genitive ẩn hoàn toàn không có "của").

Thử `pyvi`/`underthesea` (NLP tiếng Việt) trước khi chọn `nltk`+regex - bỏ vì
kéo theo build `scipy`/`scikit-learn` từ source (tự bootstrap cả `cmake`),
quá nặng cho Termux/Android, không đáng cho việc chỉ cần bắt 1-2 tiểu từ ngữ
pháp cố định.

**Kết luận chung của cả 4 cách**: độ tin cậy tỉ lệ nghịch với mức độ phải
"hiểu"/"sinh" của model - quy tắc dựa tín hiệu hình thức có thật (từ nối, POS
tag) không bao giờ tách sai (chỉ có thể bỏ sót), còn mọi mức độ để model tự
suy luận/sinh chữ đều có tỷ lệ lỗi đáng kể, kể cả khi đã thu hẹp xuống "chỉ gắn
nhãn".

## Chain: search+extract theo từng câu hỏi con (chain.py)

Nối `decompose.py` vào cơ chế đồng thuận per-link (`adaptive.py`) - `chain.py`
(độc lập, chưa nối vào `pipeline.py` chính). Câu hỏi loại bridge (có `{X}`)
thì chain tuần tự: search+extract câu hỏi con đầu, điền answer vào `{X}` của
câu hỏi con sau, search+extract tiếp. Loại comparison thì chạy 2 câu hỏi con
song song, không phụ thuộc nhau. **Chưa có bước tổng hợp câu trả lời cuối cho
câu hỏi gốc** (vd rút gọn/tính toán từ answer thô) - đó là việc của "đầu ra",
cố tình chưa làm ở giai đoạn này.

Test đầu tiên (case "thủ đô nước láng giềng phía bắc VN") lộ ngay 1 bug: answer
thô của bước 1 (đồng thuận thấp, dài dòng) vẫn bị nhét thẳng vào `{X}`, tạo
query bước 2 vỡ vụn. Vá bằng 2 việc:

1. **Cổng confidence trước khi chain**: chỉ điền `{X}` và chạy bước sau nếu
   bước trước `confident=True` (đồng thuận đã đạt ngưỡng) - không tự bịa rule
   mới, dùng thẳng tín hiệu `adaptive.py` đã có sẵn.
2. **`adaptive_entity.py`** - fork riêng của `adaptive.py`, dùng prompt mới
   `extract_entity` (pipeline.py) ép trả lời đúng 1 tên riêng thay vì 1
   câu/cụm bất kỳ. Lý do fork: bridge entity luôn là tên ngắn (nước/thành
   phố/người), còn `extract_unconditional` gốc để model tự do trích "câu/cụm
   phù hợp" nên hay lan man nhắc lại tiêu đề snippet trước khi vào nội dung -
   **đã thử chỉ cắt ngắn `n_predict` trước, không ăn thua** (chỉ cắt cụt thói
   quen lan man giữa chừng, có lúc còn tạo "đồng thuận giả" khi nhiều link
   cùng bị cắt cụt ở cùng 1 điểm do chung tiêu đề bài viết) - phải chặn từ
   prompt. Kết quả: case "cà phê" agreement bước 1 tăng 0.12 → 0.75
   (`confident=True`, ra đúng "Brazil"), tốc độ cũng nhanh hơn hẳn
   (`n_predict=20` thay vì 80).

Áp `adaptive_entity` luôn cho bước 2 khi khung ngoài cũng hỏi tên riêng (vd
"Thủ đô của {X} là gì") - route bằng `_expects_number()` (regex/từ khóa
"bao nhiêu"/"how much"/"how many"), giữ `adaptive.py` gốc khi khung ngoài hỏi
số liệu (vd "Dân số của {X} là bao nhiêu"). Case cà phê→Brazil→Brasília chạy
hết cả chain, bước 2 vẫn chưa đạt `confident=True` (0.38) nhưng agreement giờ
phản ánh đúng bất đồng nội dung thật (1 nguồn nhầm "Rio de Janeiro" - hiểu lầm
phổ biến có thật), không còn bị nhiễu do format câu trả lời khác nhau.

Case "thủ đô nước láng giềng phía bắc VN" (chạy lại với `adaptive_entity`):
đồng thuận về "Trung Quốc" tăng lên thành đa số (4/8, các link còn lại rải
rác Cam-pu-chia/Thái Lan/Việt Nam/1 answer degenerate) nhưng vẫn dưới ngưỡng
0.75 - hệ thống đúng đắn dừng lại, không chain tiếp. Đây là lần đầu
`confident=False` phản ánh đúng bất đồng thật giữa nguồn (SEO/giáo án tiếng
Việt vẫn lệch, xem phần Tune adaptive.py ở trên), không phải nhiễu do cách
trích xuất.

### Đẩy chain tới giới hạn: max_n=50, consensus_threshold=0.80

`search.py` trước chỉ lấy được ~10 kết quả (1 trang HTML của DDG, không phân
trang) - `max_n` lớn hơn 10 vô nghĩa. Thêm phân trang (theo dõi form ẩn DDG
trả kèm mỗi trang, có offset `s` + token phiên `vqd`, dừng khi đủ `n` hoặc hết
trang, giới hạn `MAX_PAGES=6` chặn vòng lặp vô hạn) - lấy được 26-30 kết quả
độc lập (đã dedupe theo url) cho các câu hỏi test, gần chạm `max_n=50` (DDG
không có đủ 50 kết quả khác nhau cho các câu hỏi này).

Chạy lại 3 câu hỏi khó có thể chain được (bridge "thủ đô nước láng giềng phía
bắc VN", comparison "Eiffel/Tokyo", comparison-theo-năm "GDP 2025 vs 2015")
với `max_n=50, consensus_threshold=0.80` (thay vì 8/0.75 mặc định):

- **Mẫu lớn hơn làm agreement giảm đúng hướng ở case vốn đã biết khó**, không
  tăng giả: case "nước láng giềng phía Bắc VN" agreement 0.50 (n=8) → 0.31
  (n=29) - `bias`/`content_bias` ở case này đều thấp (~0.11), tức bất đồng
  đến từ nguồn thật sự độc lập, không phải echo-chamber - mẫu lớn hơn đang
  phơi bày đúng bất đồng thật, không phải nhiễu định dạng. **Về mặt thống kê
  đây là dấu hiệu tốt, nhưng chỉ đúng khi lỗi giữa nguồn độc lập/ngẫu nhiên** -
  README mục "đẩy đến giới hạn" ở trên từng ghi nhận trường hợp ngược lại
  (agreement tăng sai khi lỗi mang tính hệ thống/echo-chamber, mở rộng search
  càng vớt thêm bản sao của cùng 1 lỗi gốc) - phải nhìn cả agreement lẫn
  bias/content_bias cùng nhau, không kết luận chỉ từ n hay agreement một mình.
- **Case Eiffel/Tokyo lộ ra 1 điểm về cách diễn đạt câu hỏi**: câu test dùng
  "tháp Tokyo" (không ghi "Skytree" như lần test trước) nên hội tụ đúng về
  Tokyo Tower (333m, tháp cũ, `confident=True`, agreement 0.83) thay vì Tokyo
  Skytree (634m, tháp mới) - không phải lỗi hệ thống, chỉ là `decompose.py`
  cắt nguyên văn thực thể từ câu hỏi gốc, không tự khử nhập nhằng tên. Vế
  Eiffel agreement giảm còn 0.14 (đi hết 28 link) - chiều cao bị báo lệch
  nhau nhiều giữa nguồn (300/324/325/330m, tùy có tính ăng-ten).
- **Case GDP 2025 vs 2015**: vế 2025 đạt ngưỡng nhanh (6 link, agreement 0.83,
  ra "5.026 USD" nhất quán). Vế 2015 đi hết 30 link vẫn agreement 0.27, và
  câu trả lời thắng cuộc ("8,9 triệu đồng/người/tháng") **là chỉ số khác hẳn**
  (thu nhập bình quân/tháng, không phải GDP bình quân đầu người/năm) - nguồn
  bị lẫn đơn vị/chỉ số, mở rộng max_n không cứu được vì lỗi nằm ở nội dung
  nguồn, đúng loại vấn đề "corpus lệch" đã ghi nhận trước đó, không phải bug
  cơ chế.

## Build llama.cpp từ source: bật dotprod/i8mm bị bỏ phí

Câu hỏi ban đầu: máy này (Samsung SM-S7110, chip Qualcomm SM8450/Snapdragon 8
Gen 1) có trick nào giống cách Claude Code chạy trên Termux (glibc-runner vá
ELF interpreter để chạy binary glibc trên Bionic) không? Kiểm tra thì không -
`llama-server` cài qua `pkg install` đã là native Bionic build sẵn
(`built with Clang for Android aarch64`), không có mismatch ABI nào để vá.

Nhưng tìm ra một vấn đề khác thật: gọi thẳng các hàm `ggml_cpu_has_*()`
trong `libggml-cpu.so` (qua Python `ctypes`, không cần build gì để kiểm tra)
thấy `dotprod` và `matmul_int8` (= i8mm) đều trả `0`, dù `/proc/cpuinfo` của
máy có đủ 2 feature này (`asimddp`, `i8mm`). Trên ARM, ggml không
runtime-dispatch kernel theo CPU như x86 - phải bật `-march=...+dotprod+i8mm`
lúc compile. Bản `pkg install` của Termux build generic (không chạy trên
đúng chip đích lúc build) nên không bật được.

Build lại từ source ngay trên máy (`cmake -B build -G Ninja
-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF`, giữ
`GGML_NATIVE=ON` mặc định) sửa được việc này: `GGML_NATIVE=ON` tự compile và
**chạy thử thật** một đoạn code dùng lệnh `vdotq_s32`/`vmmlaq_s32` ngay trên
CPU đang build (`check_cxx_source_runs`) - vì build trực tiếp trên chip đích,
test này pass, bật đúng `dotprod`+`i8mm`. Verify lại bằng `ctypes`:
`ggml_cpu_has_dotprod`/`_matmul_int8` chuyển từ `0` sang `1`.

Đo bằng `llama-bench` (cùng model 0.5B, cùng máy, `-t 4`):

| | pp128 (t/s) | tg64 (t/s) |
|---|---|---|
| Bản `pkg install` (generic) | 71.91 | 39.75 |
| Bản build từ source (dotprod+i8mm) | 90.75 (+26%) | 46.70 (+17%) |

`pipeline.py` trỏ thẳng `LLAMA_SERVER_BIN` sang binary build mới
(`~/vendor/llama.cpp/build/bin/llama-server`, nằm ngoài repo và ngoài
`$PREFIX` của Termux) thay vì bản trên `PATH` - không đè lên gói do `pkg`
quản lý, dễ rollback (đổi lại hằng số là xong) nếu `pkg upgrade` sau này
đổi ABI/version không tương thích.

Đã khảo sát nhưng loại bỏ vì không đáng công cho dự án này: GPU offload qua
OpenCL (Adreno 730 trên chip này chỉ "được ghi nhận là chạy được", chưa
verify chính thức - Adreno 750/830 trở lên mới được hỗ trợ chính thức; phải
ép KV-cache về f16, tắt flash-attention, dễ OOM RAM khi kết hợp `--mlock`);
Hexagon NPU (SM8450 chính thức "không được Qualcomm hỗ trợ" cho LLM NPU, dự
án cộng đồng khả dụng duy nhất dùng ExecuTorch `.pte` - khác hoàn toàn
toolchain GGUF/`llama-server` đang dùng, phải export lại model từ đầu).

## Trạng thái

Bản chạy được đầu tiên hoàn chỉnh: search + model (server warm) + benchmark
ghi log. Đã test: 8 câu hỏi đa dạng lĩnh vực với 1.5B; so sánh tốc độ 1.5B
vs 0.5B; tiếng Anh vs tiếng Việt; 8 dạng câu hỏi khác nhau ở 2 mode prompt
(synth/extract) với 0.5B; tune `adaptive.py` (per-link consensus) với 8 câu
hỏi khó hơn; khảo sát phân rã câu hỏi multi-hop trước khi search (4 cách,
chốt ở quy tắc thuần regex+POS); nối vào `chain.py` (search+extract theo
từng câu hỏi con, có cổng confidence + fork `adaptive_entity.py` cho bridge
entity, xem 2 mục trên) - chạy được, chưa nối vào pipeline chính, chưa có
bước tổng hợp câu trả lời cuối; build lại `llama-server` từ source để bật
`dotprod`/`i8mm` bị bản `pkg install` bỏ phí (+17-26% t/s, xem mục trên).

Phát hiện quan trọng nhất: **ép model nhỏ chỉ được "search, không được suy
luận" theo nghĩa chặt (mode extract) làm giảm độ tin cậy so với cho nó
tổng hợp nhẹ (mode synth)** — 0.5B không đủ khả năng theo instruction có
điều kiện phức tạp một cách ổn định. Đây ngược với giả định ban đầu của ý
tưởng, và là giới hạn thật của quy mô model, không phải vấn đề prompt.

Phát hiện thứ hai (từ tune adaptive.py): **consensus voting giữa nhiều
nguồn/lần sinh chỉ khử được lỗi ngẫu nhiên, không khử được lỗi hệ thống
lặp lại giống nhau giữa các nguồn** - "confident" không đồng nghĩa "đúng"
ở các câu multi-hop hoặc dùng kiến thức có thể lỗi thời.

Bước tiếp theo hợp lý: nối `chain.py` vào `pipeline.py` chính (câu hỏi tách
được thì chain, không tách được thì chạy như cũ); test nhánh `_expects_number`
(route câu hỏi số liệu về `adaptive.py` gốc) với 1 case bridge thật sự đủ
confident ở bước 1 để đi tới bước 2 - chưa có case nào verify được nhánh này
bằng thực nghiệm; nghĩ bước tổng hợp câu trả lời cuối cho câu hỏi gốc (đầu ra,
cố tình chưa làm); thử mode extract với model 1.5B (có tuân theo instruction
có điều kiện tốt hơn không?); chạy nhiều câu hỏi hơn để xác nhận tương quan
RAM pressure/chất lượng câu trả lời; thử câu hỏi tiếng Anh/Việt với n lớn hơn
để kết luận chắc về sự khác biệt ngôn ngữ; thử lại 2 case multi-hop/NATO còn
sai với model 1.5B để xem lỗi có phải do quy mô model hay do search/snippet
chất lượng thấp.
