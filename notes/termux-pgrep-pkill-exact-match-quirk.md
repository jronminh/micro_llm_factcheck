# Vì sao `pkill -x`/`pgrep -x` không dùng được trên máy này

`pipeline.py` (`_llama_server_pids()`) đã né vấn đề này bằng cách tự parse
`ps aux` thay vì gọi `pkill`/`pgrep`, nhưng chỉ ghi "không khớp được, đã kiểm
chứng trực tiếp" mà chưa có lý do. Đào lại và tìm được nguyên nhân gốc - ghi
lại vì đây là loại bug im lặng, dễ tưởng nhầm sang lỗi khác (đã từng làm
`_stop_server()` không dừng được server cũ mà không báo lỗi gì).

## Thực nghiệm

Test với 1 process không liên quan gì tới dự án này (`sleep`) để loại trừ
khả năng lỗi riêng của `llama-server`:

```
$ /data/data/com.termux/files/usr/bin/sleep 300 &
$ cat /proc/<pid>/comm
/data/data/com.
$ cat /proc/<pid>/cmdline | tr '\0' ' '
/data/data/com.termux/files/usr/bin/sleep 300
```

`comm` không phải basename ("sleep") mà là **15 ký tự đầu của path tuyệt đối**
dùng để exec (`/data/data/com.` = đúng 15 ký tự, khớp `TASK_COMM_LEN-1`).
Trên Linux desktop bình thường, kernel set `comm` bằng basename của file exec
(qua `kbasename()`), không phải cắt path tuyệt đối theo vị trí ký tự - đây là
khác biệt thật giữa kernel/Bionic trên máy Android này với hành vi Linux
thường gặp. Test qua cả bash background job, `nohup`, và Python
`subprocess.Popen` - kết quả giống nhau ở cả 3 cách gọi, nên không phải do
cách gọi process mà là hành vi hệ thống.

`llama-server` lại có `comm` đúng ("llama-server", không bị cắt) - vì bản
thân binary tự gọi thứ gì đó kiểu `prctl(PR_SET_NAME, ...)`/
`pthread_setname_np()` để tự đặt lại tên cho thread chính (thường thấy ở
server đa luồng, giúp `ps`/`top` dễ đọc) - ghi đè lên giá trị bị cắt ban đầu.
Đây là lý do `ps aux` (đọc `comm` sau khi đã bị ghi đè) thấy tên đúng, nhưng
không phải mọi process đều tự sửa như vậy (test `sleep` không).

**Ngay cả khi `comm` đã đúng ("llama-server"), `pgrep -x llama-server` vẫn
không khớp:**

```
$ pgrep -x llama-server          # khong ra gi, exit 1
$ pgrep -l -f llama-server
19482 /data/data/com.termux/files/home/vendor/llama.cpp/build/bin/lla
```

`pgrep -l` (kể cả khi dùng `-f` chỉ để đổi tiêu chí match) hiển thị "tên" là
1 chuỗi bị cắt cụt giữa từ ("...bin/lla"), không khớp cả với `comm` thật
("llama-server") lẫn full path đầy đủ. Tức là bản `procps-ng 3.3.17` cài qua
`pkg install` trên Termux/Android này tự đọc/parse tên process theo cách
khác - không đơn giản là đọc `/proc/PID/comm` - và bị lỗi/lệch khi làm vậy
(nghi ngờ do parser `/proc/PID/stat` dựa vào vị trí field cố định sau dấu `)`
cuối, bị lệch nếu kernel Android/vendor thêm field khác so với kernel
mainline mà procps-ng build cho Termux không tính tới - **chưa verify được
chắc chắn phần này**, chỉ verify được hành vi cuối cùng là gì).

`pgrep -f` (match theo toàn bộ cmdline, không dùng `-x`) hoạt động nhưng
**quá rộng**: sẽ khớp bất kỳ process nào có chữ "llama" ở bất kỳ đâu trong
cmdline của nó - kể cả câu lệnh `grep -i llama` hay 1 dòng lệnh shell đang
chạy đang chứa chữ đó (đã thấy thật khi test: khớp nhầm cả process
`ugrep ... llama` và `bash -c '... llama ...'` đang chạy song song).

## Kết luận thực dụng

Trên máy/bản Termux này, cả `pkill -x`/`pgrep -x` (không khớp process thật)
lẫn `-f` (khớp nhầm process không liên quan) đều không đáng tin cậy để định
vị 1 process theo tên. Cách an toàn duy nhất đã verify: tự đọc `ps aux`,
parse cột COMMAND (`ps aux` đọc `comm` đã bị ghi đè đúng, ít nhất với
process nào tự đặt tên như `llama-server`), lọc theo hậu tố đường dẫn thật
(`.endswith("/llama-server")`), rồi `kill -9 <pid>` trực tiếp - xem
`pipeline._llama_server_pids()` / `thermal.kill_llama_server()`.

Không rõ đây là lỗi của kernel Android cụ thể trên máy này, của bản dựng
`procps-ng` cho Termux, hay cả hai cộng lại - không phải hệ quả của
glibc-runner (đã loại trừ: test bằng binary Bionic thuần như `sleep`/`bash`,
không liên quan gì tới cơ chế chạy `claude` binary vá ELF). Nếu gặp lại vấn
đề tương tự trên máy khác, kiểm tra lại bằng đúng cách test ở trên
(`cat /proc/<pid>/comm` so với `pgrep -l`) trước khi kết luận.
