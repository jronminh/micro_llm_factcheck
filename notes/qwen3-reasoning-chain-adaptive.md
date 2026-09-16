# `adaptive_qwen3.py`: chain 3 bước tận dụng reasoning - thí nghiệm chưa xong

Fork của `adaptive.py` dùng riêng cho Qwen3 (xem
[[micro-llm-alternatives-to-qwen]] cho bối cảnh khảo sát model). Thay vì tắt
reasoning bằng `/no_think` như nơi khác dùng Qwen3, ý tưởng: tận dụng
reasoning để mỗi link tự lọc qua 3 bước độc lập - trích xuất → phản biện →
chốt, mỗi bước 1 system prompt riêng (`pipeline.SYSTEM_PROMPTS`
`qwen3_extract`/`qwen3_critique`/`qwen3_finalize`).

**Trạng thái: thí nghiệm chưa hoàn thiện.** Chain chạy không lỗi kỹ thuật,
nhưng bước phản biện (điểm mấu chốt của cả ý tưởng) chưa hoạt động đúng sau
2 vòng chỉnh sửa, và chi phí cao hơn baseline ~10-15 lần. Chưa nên dùng cho
pipeline chính - ghi lại để không lặp lại đường đã đi.

## Kiến trúc: đồng bộ theo bước (lockstep), không phải mỗi link tự chạy hết chain

Ban đầu mỗi link (chạy song song qua `ThreadPoolExecutor`) tự chạy hết 3
bước độc lập. Đổi sang lockstep - tất cả link trong batch xong bước 1 mới
cùng sang bước 2 - với giả thuyết: `llama-server` chọn slot dựa vào độ giống
prompt với lần trước đó trong slot đó ("LCP similarity", thấy trong
`.server.log`), nên để 1 slot xử lý liên tiếp cùng loại bước (cùng system
prompt) có thể tận dụng cache tốt hơn.

**Đã đo, giả thuyết không được ủng hộ rõ**: lockstep (71.4s) và non-lockstep
(71.5s) cho cùng 3 link gần như bằng nhau. Giữ lại lockstep vì code rõ ràng
hơn (mỗi pha có checkpoint debug riêng), không phải vì nhanh hơn.

**Song song vs tuần tự (đã đo riêng, không phụ thuộc lockstep)**: song song
luôn thắng - 71.5s (song song) so với 94.6s (tuần tự, cùng 3 link) - dù mỗi
request RIÊNG LẺ đo chậm hơn khi có cạnh tranh CPU (gen_tps ~7-14 tok/s
song song so với ~10-20 tok/s tuần tự). Máy có nhiều core thật (1+3+4
big.LITTLE) nên 3 luồng chạy đồng thời trên các core khác nhau vẫn tạo tiến
triển thật, thắng được việc mỗi luồng "nhìn có vẻ chậm hơn". Xác nhận thiết
kế gốc `N_PARALLEL=3` (từ `adaptive.py`) vẫn đúng, dù giả định ban đầu của
nó (mỗi link chỉ 1 lần gọi ngắn) không còn đúng cho chain 3 bước nặng hơn
nhiều.

## Diagnostic đã thêm vào `pipeline.run_model()`

Trả thêm `finish_reason`, `completion_tokens`, `reasoning` (nguyên văn
`reasoning_content` nếu model dùng reasoning) - dùng để debug chain nhiều
bước mà không cần đoán. `finish_reason="length"` là tín hiệu chính: model bị
cắt cụt bởi `n_predict` trước khi tự kết luận (có thể vẫn đang ở giữa
`<think>`), khác "stop" (tự dừng đúng lúc).

## Hành trình bước phản biện (critique) - chưa giải quyết xong

1. **Vòng đầu (toàn câu dễ)**: critique trả "HOP_LE" 100% số lần (12/12 lượt
   gọi qua nhiều lần test) - không có bằng chứng nó lọc được gì, vì bước
   trích xuất luôn đúng sẵn ở câu dễ.
2. **Test câu khó lần 1** ("Thủ đô của nước láng giềng phía bắc Việt Nam là
   gì?", câu đã biết cả Qwen2.5 lẫn Qwen3/no_think đều trả lời sai) - lần
   đầu thấy critique thực sự bác bỏ: 1 link trích xuất đúng "Bắc Kinh"
   (đúng!) nhưng critique kết luận "KHONG_HOP_LE - suy diễn dựa trên đoạn
   trích, không phải trích dẫn trực tiếp". **Bác bỏ nhầm câu trả lời đúng**
   vì prompt gốc không phân biệt suy luận hợp lý (từ nội dung có thật) với
   bịa đặt (nội dung không có trong đoạn trích).
3. **Sửa**: viết lại prompt `qwen3_critique` để chấp nhận suy luận hợp lý
   (vd đoạn trích nói "Trung Quốc giáp VN phía Bắc" + "thủ đô TQ là Bắc
   Kinh" → suy ra "Bắc Kinh" là HỢP LỆ), chỉ bác bỏ khi thông tin hoàn toàn
   không xuất hiện trong đoạn trích. Đồng thời tăng `n_predict_finalize`
   150→200 (bước "chốt" tưởng đơn giản vẫn bị cắt cụt ở vòng trước).
4. **Test lại cùng câu hỏi** - nhưng search (DuckDuckGo, không cố định) trả
   về **link hoàn toàn khác**, không link nào còn chứa "Bắc Kinh" nữa →
   **không kiểm chứng được** liệu bản sửa có còn bác bỏ nhầm case cũ hay
   không. Hạn chế thật của cách test này khi phụ thuộc search sống.
5. **Nhưng lộ ra vấn đề MỚI, có thể nghiêm trọng hơn**: critique giờ chấp
   nhận ("HOP_LE") cả những câu trả lời SAI rõ ràng - "Hà Nội" (đúng là thủ
   đô Việt Nam, nhưng SAI vì câu hỏi cần thủ đô của NƯỚC LÁNG GIỀNG, không
   phải VN) và "Việt Bắc" (một vùng chiến khu lịch sử, không phải quốc gia
   hay thủ đô nào - sai bản chất câu hỏi). **Nguyên nhân**: prompt chỉ kiểm
   tra "thông tin này có xuất hiện/suy ra được từ đoạn trích" (groundedness)
   - không kiểm tra "câu trả lời này có thực sự trả lời ĐÚNG TRỌNG TÂM câu
   hỏi đang hỏi hay không" (relevance). 1 trang Wikipedia về Hà Nội luôn
   khiến "Hà Nội" trông "có căn cứ" dù hoàn toàn lạc đề với câu hỏi
   multi-hop thật.

**Chưa làm**: thêm tiêu chí "phải trả lời đúng trọng tâm câu hỏi, không chỉ
đúng với nội dung đoạn trích" vào prompt `qwen3_critique`. Đây là bước tiếp
theo hợp lý nếu quay lại hướng này.

## Truncation (`finish_reason="length"`) tương quan với độ khó câu hỏi

Câu dễ: hiếm khi cắt cụt (reasoning ~400-1000 ký tự, đủ trong ngân sách).
Câu khó/mơ hồ ("không có thông tin" ở bước trích xuất): model suy luận dài
hơn hẳn (1200-1400+ ký tự) khi cố giải thích tại sao KHÔNG có thông tin hoặc
đang cân nhắc giữa nhiều khả năng - dễ chạm trần `n_predict` hơn, kể cả sau
khi tăng `n_predict_finalize`. Chưa tìm ngưỡng an toàn tuyệt đối cho câu khó.

## Chi phí đo được

~71-145s cho 3 link/1 câu hỏi tuỳ độ khó và may rủi (link timeout/dài), so
với vài giây của `adaptive.py` gốc (1 lần gọi/link, không reasoning) -
**~10-15 lần chậm hơn**. Với `max_n=8` mặc định (chưa test), chi phí cho 1
câu hỏi đầy đủ nhiều khả năng lên tới hàng trăm giây, chưa tính rủi ro
nhiệt (PMIC thường xuyên vượt 55-65°C suốt quá trình test hướng này).

## Kết luận tạm thời

Ý tưởng "dùng reasoning để tự lọc qua nhiều bước" có tiềm năng (đã thấy
critique có thể bắt lỗi thật khi tính đúng), nhưng calibration của prompt
phản biện là bài toán khó chưa giải xong - 2 lần sửa đều lộ ra lỗi mới theo
hướng khác (quá khắt khe → quá dễ dãi). Kết hợp với chi phí thời gian/nhiệt
cao hơn baseline rất nhiều lần, **chưa có lý do để thay thế `adaptive.py`
gốc bằng hướng này** ở trạng thái hiện tại. Nếu quay lại: ưu tiên sửa tiêu
chí relevance trong critique trước, và cân nhắc test trên câu hỏi cố định
sẵn (cache lại search results) thay vì search sống để so sánh các vòng sửa
prompt công bằng hơn.
