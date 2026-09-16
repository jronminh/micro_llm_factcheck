# Android hạ xung app chạy nền - dễ tưởng nhầm sang RAM/nhiệt

Ghi lại vì đây là 1 case đã đi sai hướng chẩn đoán 2 lần trước khi tìm ra
nguyên nhân thật (xem lịch sử commit `2b2a10e` -> `4909666`) - đáng nhớ để
không lặp lại: khi thấy gen_tps rớt bất thường trên máy này, kiểm tra
Termux có đang foreground/có wake-lock hay không TRƯỚC KHI nghi ngờ
RAM/nhiệt.

## Diễn biến chẩn đoán sai trước khi tìm ra đúng

1. Lần đầu thấy request chậm bất thường (outlier 20-70s+ trong khi bình
   thường vài giây) - nghi do **RAM pressure/swap** vì đang test nhiều giờ
   liên tục, `CTX_SIZE` cấp cho llama-server cũng bị nghi là nguyên nhân
   (cấp dư KV-cache). Ghi nhận trong `pipeline.py` (comment `CTX_SIZE`): "Không
   set thì llama-server mặc định dùng context tối đa của model (32768) ...
   Từng nghĩ đây là nguyên nhân chính gây swap".
2. Sau đó nghi tiếp **throttle nhiệt** (hợp lý vì test liên tục nhiều giờ).
3. Đo trực tiếp mới ra nguyên nhân thật: **gen_tps rơi từ ~27 tok/s xuống
   0.17 tok/s** - mức sụt quá lớn (>150 lần) để giải thích bằng RAM/nhiệt
   thông thường, chỉ khớp với 1 cơ chế: Android (OneUI) chủ động hạ xung
   nhịp CPU cụm "big" (performance core) cho app không ở foreground, kể cả
   khi app đó (Termux) đang có tiến trình nền chạy nặng.

## Cơ chế thật

Android có nhiều lớp quản lý năng lượng cho app nền (Doze mode, App Standby
Buckets, cgroup cpuset khác nhau cho foreground/background) - khi Termux
không phải app đang hiển thị/tương tác, hệ thống có thể chuyển tiến trình
của nó (và mọi process con như `llama-server`) sang cụm core "small"
(efficiency core), bất kể process đó đang thực sự bận CPU 100%. Đây không
phải nhiễu ngẫu nhiên, mà là chính sách tiết kiệm pin có chủ đích của OneUI.

## Cách khắc phục đã áp dụng

`termux-wake-lock` (lệnh từ gói `termux-api`) giữ CPU không bị đưa vào chế
độ Doze/background throttle. `pipeline.ensure_server()` gọi lệnh này tự động
mỗi lần (an toàn nếu gọi lại khi đã lock rồi, không có tác dụng phụ):

```python
subprocess.run(["termux-wake-lock"], check=False)
```

Sau khi thêm, tốc độ ổn định trở lại (không còn outlier kiểu 0.17 tok/s).

## Bài học chẩn đoán

Sụt tốc độ generation kiểu "rớt hàng chục đến hàng trăm lần" trên
Termux/Android gần như chắc chắn là background throttle, không phải RAM
hay nhiệt (RAM pressure/swap thường làm chậm dần đều hoặc timeout hẳn, nhiệt
throttle thường chỉ giảm vài chục % chứ không giảm 2-3 bậc độ lớn). Kiểm tra
`termux-wake-lock` đã bật và Termux có đang chạy foreground hay không trước
khi đào sâu hướng khác.

Liên quan: [[android-thermal-apis]] (nhiệt độ CPU-die 80-90°C dưới tải nặng
là *khác* hiện tượng throttle-vì-chạy-nền này - 2 cơ chế độc lập, đều có thể
ảnh hưởng tốc độ nhưng theo cách khác nhau).
