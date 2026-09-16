# Android/Termux: đọc nhiệt độ đúng cách

Ghi lại vì đã đi sai hướng 1 lần khi làm kill switch nhiệt cho
`compare_binaries.py`/`guarded_run.py` (xem `thermal.py`) - nếu sau này cần
đọc nhiệt độ máy cho việc khác, đọc note này trước khi chọn cảm biến.

## Hai lớp API nhiệt độ trên Android, không phải một

1. **`/sys/class/thermal/thermal_zone*/temp`** (sysfs) - số đo thô
   (millidegree C, chia 1000 ra độ C) thẳng từ từng cảm biến phần cứng.
   Không có diễn giải mức độ nguy hiểm nào, chỉ là con số. File `type` cùng
   thư mục cho biết đang đo cái gì (`cat thermal_zoneN/type`). Đọc được thẳng
   từ Termux, không cần quyền gì đặc biệt.
2. **Android Thermal API chính thức** (`PowerManager.getCurrentThermalStatus()`
   / `IThermalService`, xem [Android thermal mitigation
   docs](https://source.android.com/docs/core/power/thermal-mitigation) và
   [Thermal API cho game](https://developer.android.com/games/optimize/adpf/thermal)) -
   lớp OS tổng hợp nhiều cảm biến + skin temp model thành 1 mức độ throttling
   chuẩn hoá: `NONE → LIGHT → MODERATE → SEVERE → CRITICAL → EMERGENCY →
   SHUTDOWN`. Đây là tín hiệu "đúng nghĩa nguy hiểm" mà app Android dùng để
   tự hạ tải, nhưng cần `dumpsys thermalservice` hoặc JNI - **đã thử,
   `dumpsys` không có trong `$PATH` của Termux**, sandbox không cho truy cập
   lớp này. Nếu sau này cần dùng, phải tìm cách khác (root, hoặc Termux:API
   extension nếu có), chưa tìm ra cách nào từ Python thuần trong Termux.

Kết luận thực dụng: trong Termux, chỉ có lớp 1 (sysfs) khả dụng. Phải tự diễn
giải con số thô, Android sẽ không làm hộ việc đó.

## Không phải zone nào cũng đo cùng một thứ

Trên máy này (Samsung SM-S7110, Snapdragon 8 Gen 1 / SM8450), quét
`/sys/class/thermal/thermal_zone*/type` ra 63 zone, đáng chú ý:

- `cpuss-0..3`, `cpu-0-0..3`, `cpu-1-0..8` - cảm biến **junction ngay tại
  nhân CPU** (per-core + cluster). Đây là nhiệt độ die thực, không phải nhiệt
  độ máy nói chung.
- `pm8350_tz`, `pm8350c_tz` - cảm biến trên **PMIC** (chip quản lý nguồn,
  thường đặt gần pin/mainboard). Bám sát nhiệt độ pin và "máy nóng khi cầm
  tay" hơn nhiều so với die CPU.
- `gpuss-0/1`, `mdmss-*`, `camera-*`, `ddr`, `video`... - các subsystem khác,
  không liên quan tới suy luận LLM (CPU-bound).
- Rất nhiều zone modem/mmWave (`mmw*`, `sub1-*`, `sdr*`) trả `-273000` (tuyệt
  đối không, tức không có cảm biến/không active) - phải lọc bỏ giá trị vô lý
  khi tính max.

Số đo song song lúc máy đang chạy pipeline nặng (per-link consensus,
`explore_adaptive.py`) so với lúc nghỉ:

| Zone | Lúc nghỉ | Lúc tải nặng |
|---|---|---|
| `cpuss-0..3` / `cpu-*` (CPU die) | ~40-44°C | **86-93°C** |
| `pm8350_tz` / `pm8350c_tz` (PMIC) | ~37-41°C | ~50-60°C (tăng chậm hơn nhiều) |
| Pin (`termux-battery-status`, độ chính xác 0.1°C) | ~34-38°C | chưa đo tới lúc rất nóng |

## Vì sao ban đầu chọn sai ngưỡng

Lần đầu làm kill switch, lấy max của các zone `cpuss-*`/`cpu-0-*`/`cpu-1-*`
với ngưỡng 60°C - sai lầm: đây là cảm biến **junction CPU**, chạy lên 80-95°C
dưới tải nặng là **bình thường trên hầu hết SoC di động hiện đại**, không
phải dấu hiệu nguy hiểm. TJmax (ngưỡng phần cứng tự shutdown) trên các chip
cỡ này thường ~100-115°C, và bản thân SoC đã có governor DVFS tự hạ xung
*trước khi* chạm ngưỡng đó - phần cứng tự bảo vệ nó độc lập với bất kỳ script
nào ở tầng Python. Hậu quả: ngưỡng 60°C trên CPU-die kích hoạt gần như ngay
lập tức ở bất kỳ tác vụ suy luận nào (thấy 3 lần liên tiếp, luôn <15s để chạm
60°C) - không phân biệt được "máy đang nóng thật sự nguy hiểm" với "CPU đang
được dùng bình thường".

`pm8350_tz`/`pm8350c_tz` (PMIC) mới là chỉ số bám sát nghĩa thường ngày của
"máy nóng quá" (cầm tay khó chịu, ảnh hưởng pin) - tăng chậm hơn nhiều và ở
mức thấp hơn hẳn ngay cả khi CPU-die đang ở 90°C.

## Thiết kế đã áp dụng (xem `thermal.py`)

Kill switch 2 tầng:
- **PMIC là tín hiệu chính**: ngưỡng 50°C (ban đầu 45°C, nâng lên sau khi
  thấy máy thường xuyên chạm 45°C ngay cả ở tải vừa phải, gây chờ nguội/dừng
  quá thường xuyên so với mức rủi ro thật), liên tục ≥180s → dừng. Đây là
  điều kiện thực sự phản ánh "máy nóng đáng lo".
- **CPU-die là failsafe phụ**: ngưỡng 100°C (sát TJmax), liên tục ≥60s → dừng.
  Chỉ nên kích khi có gì thật bất thường (governor phần cứng không kịp phản
  ứng), không phải điều kiện vận hành bình thường.
- **Ngưỡng chờ nguội trước khi bắt đầu** (`COOLDOWN_BEFORE_START_C`): 45°C
  (ban đầu 40°C, nâng lên vì 40°C khiến các lần test liên tiếp phải chờ
  nguội lâu không cần thiết) - đợi PMIC xuống dưới mức này mới khởi động
  server cho lần test mới.

Nguồn tham khảo:
- [Thermal mitigation | Android Open Source Project](https://source.android.com/docs/core/power/thermal-mitigation)
- [Thermal API | Android game development | Android Developers](https://developer.android.com/games/optimize/adpf/thermal)
- [sysfs-class-thermal ABI (kernel.org)](https://www.kernel.org/doc/Documentation/ABI/testing/sysfs-class-thermal)
