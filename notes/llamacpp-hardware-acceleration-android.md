# Tăng tốc phần cứng cho llama.cpp trên Android/Termux: cái nào đáng làm

Tổng hợp lại nghiên cứu đã làm khi tìm cách chạy nhanh hơn trên chip này
(Samsung SM-S7110, Qualcomm SM8450/Snapdragon 8 Gen 1) - 1 hướng đã làm và
verify được lợi ích thật, 2 hướng đã khảo sát rồi loại bỏ có chủ đích.

## Đã làm: bật dotprod/i8mm bằng build từ source

(Xem thêm `pipeline.py` comment `LLAMA_SERVER_BIN` và README mục "Build
llama.cpp từ source" - đây là bản tổng hợp lại, không thay thế.)

Câu hỏi ban đầu: máy này có cần trick kiểu glibc-runner (vá ELF interpreter
để chạy binary glibc trên Bionic, cách Claude Code chạy trên Termux) để chạy
llama.cpp không? Kiểm tra bằng `file`/version string - không cần, bản
`pkg install` của Termux đã là native Bionic build sẵn, không có mismatch
ABI nào để vá.

Nhưng gọi thẳng `ggml_cpu_has_dotprod()`/`ggml_cpu_has_matmul_int8()` (=i8mm)
qua Python `ctypes` trên `libggml-cpu.so` (không cần build gì để kiểm tra)
thấy cả 2 đều trả `0`, dù `/proc/cpuinfo` của máy có đủ `asimddp`/`i8mm`.

**Lý do**: trên ARM, `ggml` không runtime-dispatch kernel theo CPU feature
như trên x86 (không có kiểu "compile nhiều phiên bản, chọn phiên bản phù hợp
lúc chạy") - phải bật cờ compile-time `-march=...+dotprod+i8mm`. Bản
`pkg install` của Termux build generic (không chạy trên đúng chip đích lúc
build gói) nên không tự bật được các cờ này.

**Cách sửa**: build lại từ source ngay trên máy đích (không cross-compile),
giữ `GGML_NATIVE=ON` (mặc định của CMake build llama.cpp) - flag này tự
compile và **chạy thử thật** 1 đoạn code dùng lệnh `vdotq_s32`/`vmmlaq_s32`
ngay trên CPU đang build (`check_cxx_source_runs` của CMake) - vì build
trực tiếp trên chip đích, test pass, tự bật đúng `-march=...+dotprod+i8mm`.
Verify lại bằng `ctypes`: cả 2 hàm chuyển từ `0` sang `1`.

Đo bằng `llama-bench` (model 0.5B, `-t 4`, cùng máy):

| | pp128 (t/s) | tg64 (t/s) |
|---|---|---|
| Bản `pkg install` | 71.9 | 39.75 |
| Bản build từ source | 90.8 (+26%) | 46.7 (+17%) |

Về sau đo lại trong pipeline thật (không phải `llama-bench` tho, xem
`compare_binaries.py`) - lợi ích còn rõ hơn (nhanh gấp ~2-3.5 lần trên câu
hỏi ngắn), có thể vì bản `pkg` còn chịu thêm overhead khác ngoài phần
dotprod/i8mm mà `llama-bench` không đo hết.

### Cơ chế cụ thể: vì sao dotprod/i8mm tăng tốc

Phần tốn compute nhất khi suy luận là nhân ma trận trọng số đã lượng tử hoá
(Q4_K_M - trọng số 4-bit + scale theo block) với vector activation (lượng
tử hoá động sang int8 lúc chạy) - về bản chất là hàng triệu phép nhân-cộng
số nguyên 8-bit lặp lại.

`dotprod` và `i8mm` là 2 tập lệnh CPU tăng tốc đúng phép toán đó:

- **`dotprod`** (ARMv8.2, lệnh `SDOT`/`UDOT`, intrinsic `vdotq_s32`): 1 lệnh
  CPU tính luôn tích vô hướng của 4 cặp số int8 rồi cộng dồn vào thanh ghi
  int32 - thay vì 4 lệnh nhân + 4 lệnh cộng riêng lẻ.
- **`i8mm`** (ARMv8.6, lệnh `SMMLA`/`UMMLA`, intrinsic `vmmlaq_s32`): tiến
  thêm 1 bậc - 1 lệnh làm luôn cả phép nhân ma trận nhỏ (block 8×8 int8),
  gộp được nhiều dotprod vào 1 lệnh CPU duy nhất.

Cùng một khối lượng phép nhân-cộng, CPU cần **ít lệnh hơn hẳn** để hoàn
thành - không đổi thuật toán, không đổi độ chính xác, không bỏ bớt phép
tính nào, chỉ dùng lệnh CPU mạnh hơn xử lý nhiều dữ liệu hơn mỗi lệnh.

**Vì sao phải build lại mới bật được, không tự động có sẵn**: khác x86
(nhiều thư viện làm function multiversioning - compile sẵn nhiều phiên bản,
tự chọn lúc chạy theo CPU thật), `ggml` **không runtime-dispatch trên ARM**
- việc dùng `dotprod`/`i8mm` quyết định bằng `#ifdef` ngay lúc compile qua
cờ `-march=...+dotprod+i8mm`, không thể bật/tắt lúc chạy. Bản `pkg install`
build 1 lần cho mọi thiết bị ARM khác nhau nên build generic, không dám giả
định máy đích có 2 feature này - tắt hết, rơi về nhánh chậm hơn.
`GGML_NATIVE=ON` giải quyết bằng cách build ngay trên máy sẽ chạy nó: không
chỉ đọc `/proc/cpuinfo` để đoán, mà thực sự **compile và chạy thử** đoạn
code dùng `vdotq_s32`/`vmmlaq_s32` ngay trên CPU đang build
(`check_cxx_source_runs` của CMake) - build trực tiếp trên chip đích thật
nên việc bật cờ chính xác 100%, không phải đoán.

**Vì sao pp (prompt eval) tăng nhiều hơn tg (token generation)**: pp128
+26% > tg64 +17%. Prompt eval xử lý cả loạt token cùng lúc → nhân ma trận
theo batch lớn, tận dụng `i8mm` (mạnh nhất khi có nhiều dữ liệu để gộp vào
1 lệnh) rõ rệt hơn. Token generation sinh từng token một, mỗi bước chỉ có 1
vector nhỏ để nhân - ít cơ hội gộp lệnh hơn, và bắt đầu bị giới hạn bởi băng
thông đọc bộ nhớ (đọc trọng số từ RAM) nhiều hơn là tốc độ tính toán thuần -
đây là nút thắt khác (memory-bound, không phải compute-bound) mà lượng tử
hoá KV-cache (`--cache-type-k`/`-v`, chưa test - xem mục "hướng chưa khai
thác" nếu có ghi ở README/thảo luận) mới nhắm đúng vào.

**Bài học tổng quát**: với bất kỳ binary ARM nào build sẵn qua package
manager (không riêng llama.cpp), luôn đáng nghi ngờ có bỏ phí ISA extension
của chip đích - kiểm tra bằng cách gọi thẳng hàm "has_feature" của thư viện
(nếu có) qua `ctypes`/tương đương, không cần build lại để kiểm tra trước.

## Đã khảo sát, loại bỏ: GPU offload qua OpenCL

Adreno 730 (GPU trên SM8450) chỉ "được cộng đồng ghi nhận là chạy được" với
backend OpenCL của llama.cpp, **chưa được verify chính thức** - theo tài
liệu/issue tracker của llama.cpp, chỉ Adreno 750/830 trở lên mới nằm trong
danh sách hỗ trợ chính thức. Ngay cả nếu chạy được, backend OpenCL của
llama.cpp bắt buộc ép KV-cache về f16 và tắt flash-attention (2 giới hạn
kỹ thuật của backend, không phải tuỳ chọn) - dễ OOM RAM khi kết hợp
`--mlock` (khoá RAM không cho swap ra, đã dùng trong pipeline này cho CPU
path). Kết luận: rủi ro/công sức không xứng đáng cho dự án ở quy mô này -
không làm.

## Đã khảo sát, loại bỏ: Hexagon NPU

SM8450 chính thức "không được Qualcomm hỗ trợ" cho suy luận LLM qua NPU
(Hexagon). Dự án cộng đồng khả dụng duy nhất dùng ExecuTorch (định dạng
`.pte`) - khác hoàn toàn toolchain GGUF/`llama-server` đang dùng trong dự án
này, nghĩa là phải export lại model từ đầu sang định dạng khác, không tái
dùng được pipeline hiện tại. Không làm vì chi phí chuyển đổi toolchain quá
lớn so với lợi ích chưa chắc chắn (NPU không chính thức hỗ trợ chip này).

## Tổng kết quyết định

| Hướng | Trạng thái | Lý do |
|---|---|---|
| CPU: dotprod/i8mm qua build from source | **Đã làm, dùng mặc định** | Verify được lợi ích thật (+17-26% llama-bench, hơn nữa trong pipeline thật), chi phí thấp (build 1 lần, không đổi toolchain) |
| GPU: OpenCL/Adreno | Loại bỏ | Chip chưa hỗ trợ chính thức, kèm giới hạn kỹ thuật (f16 KV-cache, không flash-attention) dễ OOM |
| NPU: Hexagon/ExecuTorch | Loại bỏ | Chip không được hỗ trợ chính thức, phải đổi hẳn toolchain (export lại model) |
