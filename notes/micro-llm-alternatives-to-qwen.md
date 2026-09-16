# Khảo sát model nhỏ khác ngoài Qwen2.5 (nghiên cứu web, tháng 9/2026)

Dự án đang dùng Qwen2.5-0.5B/1.5B-Instruct (Q4_K_M). Phát hiện cốt lõi đã
ghi trong README: **model nhỏ không theo tốt instruction có điều kiện phức
tạp** (mode extract kém tin cậy hơn synth ở 0.5B). Khảo sát này tìm xem có
model nào cùng lớp kích thước nhưng theo instruction tốt hơn - trước khi
nghĩ tới việc phải tăng size (đổi lấy chậm hơn/nóng hơn, đã đo được chi phí
thật ở nhánh nhiệt độ đang làm).

**Lưu ý**: kiến thức nền của model dừng ở đầu 2026, đã search web bổ sung
cho phần "hiện tại" (9/2026) - các con số benchmark dưới đây trích từ nguồn
thứ 3 (blog/HuggingFace), chưa tự đo lại trên máy này.

## Ứng viên đáng thử trước, cùng lớp kích thước hiện tại

### Qwen3-0.6B / Qwen3-1.7B - ứng viên hàng đầu

Cùng nhà (Qwen), cùng cỡ tham số với model đang dùng (0.5B/1.5B), nên gần
như chắc chắn tương thích prompt template/tokenizer convention hiện có -
chi phí thử nghiệm thấp nhất trong các lựa chọn.

- Theo Qwen team: **Qwen3-1.7B-Base đạt hiệu năng ngang Qwen2.5-3B-Base**
  (cải tiến kiến trúc + training, không chỉ do scale tham số).
  Qwen3 dense base models ở nhiều size còn vượt cả model Qwen2.5 lớn hơn 1
  bậc (Qwen3-4B > Qwen2.5-7B trong 1 số benchmark).
- Qwen team nói rõ Qwen3 **"excel in instruction following"** - đúng chỗ
  yếu đã đo được của Qwen2.5-0.5B trong dự án này.
- GGUF chính chủ có sẵn (`Qwen/Qwen3-0.6B-GGUF`), quant Q8_0 ~639MB.

**Cảnh báo cần verify trước khi thử**: từng có bug
`ValueError: Architecture qwen3 not supported` trên llama.cpp (issue
`ggml-org/llama.cpp#13157`) ở các bản cũ - cần xác nhận bản llama.cpp build
từ source trong dự án này (build gần đây, xem `LLAMA_SERVER_BIN`) đã đủ mới
để support kiến trúc `qwen3` chưa, trước khi tải model về test (`llama-cli
--version` hoặc thử load thẳng, không cần đoán).

### Gemma 3 270M - ứng viên cực nhỏ, khác hướng

Nhỏ hơn cả Qwen2.5-0.5B đang dùng (270M tham số) nhưng theo Google, được
**thiết kế riêng để mạnh về instruction-following dù kích thước cực nhỏ**
("thiết lập mức hiệu năng mới cho kích thước của nó" theo IFEval). Định
hướng khác Qwen2.5 (vốn là general-purpose scale-down, không tối ưu riêng
cho instruction-following ở size nhỏ).

- Chạy được trên Android qua llama.cpp, benchmark cộng đồng ghi nhận tiêu
  thụ pin cực thấp (<1% pin cho nhiều lượt hội thoại) - hợp với ràng buộc
  nhiệt/pin của dự án này hơn hẳn so với đi lên 3B.
- GGUF chính thức có (`ggml-org/gemma-3-270m-GGUF`, kèm bản QAT - quantize-
  aware training, thường giữ chất lượng tốt hơn quantize thường ở cùng
  bit-width).
- Rủi ro: 270M rất nhỏ, có thể không đủ để tổng hợp mạch lạc (README đã ghi
  "dưới 1B thường quá yếu để tổng hợp mạch lạc" - nhưng đó là kết luận rút
  ra TỪ Qwen2.5, chưa chắc đúng với model được tối ưu riêng cho
  instruction-following như Gemma 3 270M). Đáng thử vì chi phí thử rất
  thấp (model bé, tải/test nhanh).

## Ứng viên lớn hơn - cân nhắc sau, đổi lấy nhiều nhiệt/thời gian hơn

- **SmolLM3-3B** (Hugging Face, Apache 2.0): theo benchmark công bố, vượt
  Llama-3.2-3B và Qwen2.5-3B, cạnh tranh được với vài model 4B. Ở đúng biên
  trên (3B) mà README đã đặt ra cho dự án này - nếu thử, cần đo lại thời
  gian/nhiệt kỹ vì đã biết 1.5B hiện tại đã đẩy máy lên vùng nóng.
- **Phi-4-mini (3.8B)**: vượt ngưỡng 3B đã đặt ra cho dự án, không ưu tiên
  trừ khi các lựa chọn nhỏ hơn đều không đạt.

## Bối cảnh rộng hơn (ghi nhận, chưa đào sâu)

Tính tới 9/2026, hệ sinh thái SLM còn có Gemma 4 (E2B/E4B/12B), Qwen3.5
Small (0.8B-9B) - các dòng mới hơn cả Qwen3/Gemma 3 nói trên. Chưa khảo sát
chi tiết (ngoài phạm vi lần này) - nếu Qwen3/Gemma 3 270M không đạt, đây là
điểm bắt đầu tiếp theo.

## Cập nhật sau khi tải về và test thật (9/2026)

Đã tải `Qwen3-0.6B-Q4_K_M` (bartowski) và `gemma-3-270m-it-Q4_K_M` (unsloth,
lưu ý: repo chính chủ `ggml-org/gemma-3-270m-GGUF` chỉ có bản **base**,
phải lấy bản `-it` từ community re-quant khác). llama.cpp build từ source
(commit `d1d3c33`, 15/9/2026) load được cả 2 kiến trúc không lỗi - bug
"Architecture qwen3 not supported" (issue #13157) đã được fix từ lâu, không
còn là vấn đề.

### Qwen3-0.6B: bug tương thích thật với pipeline hiện tại, đã tìm ra cách sửa

Chạy `compare_models.py` (3 câu hỏi đơn giản, có search snippet thật):
**2/3 câu trả lời rỗng**. Probe raw JSON response (`scripts/_probe_raw.py`)
lộ ra nguyên nhân: Qwen3 là model "dual-mode reasoning", mặc định tự bật
chế độ suy luận `<think>...</think>` - `llama-server` tách nội dung này ra
field riêng `reasoning_content`, khác với `content` (câu trả lời cuối) mà
`pipeline.run_model()` đang đọc. **Cả 2 field dùng chung 1 ngân sách token**
(`max_tokens`) - với câu khó/dài, model dùng hết ngân sách để suy luận mà
chưa kịp đóng thẻ `<think>` để bắt đầu sinh `content`, kết quả `content`
rỗng dù model đã "làm việc" thật (thấy trong 1 case: 125 completion_tokens
sinh ra hoàn toàn nằm trong `reasoning_content` bằng tiếng Anh, dù input
tiếng Việt).

**Cách sửa đã verify hoạt động**: thêm `/no_think` vào cuối user prompt
(cách chính thức Qwen team khuyến nghị để tắt reasoning per-turn, không cần
đổi chat template). Test lại cùng câu hỏi: `completion_tokens` giảm từ 125
xuống 11, không còn `reasoning_content`, `content` trả lời đúng ngay. Nếu
quyết định dùng Qwen3, `pipeline.py` cần thêm logic tự nối `/no_think` khi
`MODEL_PATH` là model Qwen3 - **chưa làm**, vì dự án chưa quyết định chuyển
hẳn sang Qwen3, chỉ mới khảo sát.

Hướng ngược lại - tận dụng reasoning thay vì tắt nó - đã thử riêng ở
`adaptive_qwen3.py`, xem [[qwen3-reasoning-chain-adaptive]] (thí nghiệm
chưa hoàn thiện, chi phí cao hơn baseline ~10-15 lần, chưa vượt qua được
`adaptive.py` gốc).

### Gemma3-270M: vi phạm ràng buộc grounding, không phải bug kỹ thuật

Probe cố tình gọi model **không kèm đoạn trích nào**, system prompt vẫn yêu
cầu rõ "chỉ trả lời dựa trên đoạn trích, nếu không đủ thông tin thì nói rõ
không có thông tin". Kết quả đối chiếu ngay trong cùng điều kiện:

- Qwen3 tuân thủ đúng: trả lời "không có thông tin."
- **Gemma3-270m tự tin bịa** "Chiều cao núi Everest là 8.844 mét." - sai số
  liệu, và phớt lờ hoàn toàn chỉ dẫn refuse-khi-thiếu-thông-tin.

Đây là vấn đề nghiêm trọng hơn dự đoán ban đầu ("có thể không đủ để tổng
hợp mạch lạc") - không chỉ yếu, mà **chủ động vi phạm đúng ràng buộc cốt
lõi của cả dự án** (chỉ được search, không được trả lời từ tri thức nội
tại). Cũng quan sát thêm: trong `compare_models.py` (có search snippet
thật), 2/3 câu hỏi khác của Gemma3-270m trả về `gen_tps=0.0`/rỗng hoàn
toàn (dừng sinh ngay lập tức) - nguyên nhân riêng, chưa điều tra, không
liên quan tới bug thiếu context ở trên.

**Đánh giá tạm thời**: Gemma3-270M rủi ro cao cho mục tiêu grounding chặt
của dự án này, cần thêm bằng chứng trước khi đầu tư thêm công sức (vd sửa
prompt/system instruction mạnh hơn, hoặc chấp nhận loại bỏ khỏi danh sách
ứng viên).

## Đề xuất thứ tự thử (đã cập nhật sau lần test đầu)

1. ~~Verify llama.cpp support kiến trúc `qwen3`~~ - **xong, hoạt động tốt**.
2. ~~Tải về, chạy sanity test~~ - **xong, xem mục "Cập nhật" ở trên**.
3. **Tiếp theo**: chạy lại `explore_formats.py`/`explore_adaptive.py` (bộ
   câu hỏi khó cũ) cho Qwen3-0.6B **với `/no_think` đã áp dụng** - cần sửa
   tạm cách gọi (nối `/no_think` vào cuối câu hỏi trước khi qua
   `answer_question_per_link`, hoặc patch `build_user_prompt` tạm thời cho
   lần test này) để so sánh công bằng với Qwen2.5-0.5B (cùng size, không bị
   nhiễu bởi bug reasoning-budget).
4. Gemma3-270M: cân nhắc dừng khảo sát thêm trừ khi có lý do cụ thể để tiếp
   tục - vi phạm ràng buộc grounding cốt lõi ngay ở test đầu tiên, không
   phải vấn đề có thể sửa bằng prompt engineering nhẹ.
5. Chỉ cân nhắc SmolLM3-3B/Phi-4-mini nếu Qwen3-0.6B (sau khi sửa) vẫn
   không đạt yêu cầu instruction-following, và chấp nhận đánh đổi thêm
   nhiệt/thời gian.

Sources:
- [Qwen3: Think Deeper, Act Faster (Qwen blog)](https://qwenlm.github.io/blog/qwen3/)
- [Qwen/Qwen3-0.6B-GGUF (Hugging Face)](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF)
- [llama.cpp Architecture qwen3 not supported (GitHub issue #13157)](https://github.com/ggml-org/llama.cpp/issues/13157)
- [Introducing Gemma 3 270M (Google Developers Blog)](https://developers.googleblog.com/en/introducing-gemma-3-270m/)
- [gemma-3-270m-qat GGUF (Hugging Face)](https://huggingface.co/ggml-org/gemma-3-270m-qat-GGUF)
- [The Best Open-Source Small Language Models (SLMs) in 2026 (BentoML)](https://www.bentoml.com/blog/the-best-open-source-small-language-models)
- [10 Small Language Models to Know in 2026 (Turing Post)](https://www.turingpost.com/p/slmslist)
