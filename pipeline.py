"""Search + model, ghép prompt ép model chỉ tổng hợp trên snippet đã search được.

Model chạy qua llama-server (HTTP, OpenAI-compatible). Lý do không dùng
llama-cli trực tiếp: ở chat/conversation mode, llama-cli echo lại prompt vào
stdout xen với câu trả lời (và cắt ngắn phần echo khi prompt dài), không có
ranh giới rõ ràng để tách câu trả lời thật ra bằng regex. llama-server trả
JSON sạch, đồng thời giữ model warm giữa các lần gọi.
"""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

from search import search

MODEL_PATH = Path(__file__).parent / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
SERVER_URL = "http://127.0.0.1:8080"
SERVER_LOG = Path(__file__).parent / ".server.log"

# Bản llama-server cài qua `pkg install` được build generic (không bật
# +dotprod+i8mm dù CPU máy này có, xác nhận qua ggml_cpu_has_dotprod()/
# _matmul_int8() = 0) vì package build không chạy trên chính con chip đích
# nên không auto-detect được (xem README). Build tay từ source ngay trên máy
# (GGML_NATIVE=ON tự chạy thử lệnh dotprod/i8mm thật trên CPU) bật được cả
# hai - đo bằng llama-bench: pp128 71.9->90.8 t/s (+26%), tg64 39.8->46.7 t/s
# (+17%), cùng model, cùng máy. Dùng thẳng đường dẫn build này thay vì bản
# pkg trên PATH.
LLAMA_SERVER_BIN = Path.home() / "vendor" / "llama.cpp" / "build" / "bin" / "llama-server"

# Không set thì llama-server mặc định dùng context tối đa của model (32768 với
# Qwen2.5), cấp KV-cache cho toàn bộ 32k token dù prompt thật chỉ ~200-400
# token (1 snippet + n_predict=80). Từng nghĩ đây là nguyên nhân chính gây
# swap - thực ra gốc rễ thọt nặng là OneUI hạ cpuset khi Termux chạy nền
# (xem README), nhưng vẫn giữ ctx nhỏ vì không có lý do gì cấp dư.
CTX_SIZE = 3072

# Số slot song song trong llama-server - adaptive.py dùng để xử lý nhiều link
# đồng thời (thay vì tuần tự) khi cần nhiều link/vote để tune với câu hỏi khó.
# Với foreground + wake-lock, CPU không còn là nút cổ chai chính nên tăng lên
# vẫn an toàn; CTX_SIZE=3072 chia cho 3 slot còn ~1024 token/slot, đủ nhiều
# so với prompt thật.
N_PARALLEL = 3

# "synth": model được phép ghép câu, miễn bám sát snippet (như trợ lý hỏi-đáp).
# "extract": model chỉ được copy nguyên văn một đoạn trong snippet, không diễn
# giải, không ghép câu — ép nó hoạt động như một công cụ tìm-và-trích, không
# phải một model đang "trả lời".
SYSTEM_PROMPTS = {
    "synth": (
        "Bạn chỉ được trả lời dựa trên các đoạn trích dưới đây, không dùng kiến thức "
        "nào khác. Khi trả lời, trích dẫn đúng câu chữ trong đoạn trích liên quan. "
        "Nếu các đoạn trích không đủ thông tin để trả lời, nói rõ \"không có thông tin\"."
    ),
    "extract": (
        "Bạn là một công cụ tìm kiếm, không phải trợ lý hội thoại. Nhiệm vụ duy nhất: "
        "tìm trong các đoạn trích dưới đây một câu hoặc cụm từ NGUYÊN VĂN (copy chính "
        "xác từng chữ, không viết lại, không thêm từ nối, không giải thích) trả lời "
        "trực tiếp yêu cầu. Chỉ in ra đúng đoạn đó, không gì khác. Nếu không có đoạn "
        "nào phù hợp, in ra đúng một từ: KHONG_CO."
    ),
    # Bỏ nhánh "nếu không có thì in KHONG_CO" - README ghi nhận model nhỏ theo
    # instruction có điều kiện không ổn định. Quyết định "có đủ thông tin hay
    # không" chuyển sang Python, qua đồng thuận giữa nhiều lần sinh (adaptive.py).
    "extract_unconditional": (
        "Bạn là một công cụ tìm kiếm, không phải trợ lý hội thoại. Nhiệm vụ duy nhất: "
        "tìm trong các đoạn trích dưới đây một câu hoặc cụm từ NGUYÊN VĂN (copy chính "
        "xác từng chữ, không viết lại, không thêm từ nối, không giải thích) trả lời "
        "trực tiếp yêu cầu. Chỉ in ra đúng đoạn đó, không gì khác."
    ),
    # Thử nghiệm: thay vì copy nguyên văn 1 mảnh trích (độ dài/hình dạng khác
    # nhau tùy đoạn - "324 mét" ngắn gọn vs "Tháp Eiffel sở hữu cấu trúc mạng
    # lưới sắt độc đáo, cao khoảng 324 mét..." dài dòng - làm cluster hóa theo
    # câu (SequenceMatcher) không nhận ra chúng cùng giá trị), ép model viết
    # lại thành 1 câu khẳng định ngắn, chuẩn hóa (chủ ngữ-vị ngữ-giá trị).
    # Hai lợi ích: (1) các answer cùng giá trị nay có hình dạng câu giống nhau
    # hơn, cluster hóa theo chuỗi đáng tin hơn; (2) so được "khung câu" của
    # answer với "khung câu" của câu hỏi (xem adaptive._claim_match) để bắt
    # case answer bám đúng chủ đề nhưng lệch sự kiện con (vd "Eiffel cao thêm
    # 15cm mùa hè" khi hỏi tổng chiều cao) - điều mà so từ khóa thô không bắt
    # được vì 2 câu chia sẻ phần lớn từ vựng, chỉ khác phần vị ngữ/giá trị.
    "extract_claim": (
        "Bạn là một công cụ tìm kiếm, không phải trợ lý hội thoại. Nhiệm vụ duy nhất: "
        "đọc đoạn trích dưới đây, tìm sự kiện cụ thể trả lời trực tiếp yêu cầu, rồi viết "
        "lại thành ĐÚNG MỘT câu khẳng định ngắn, đủ chủ ngữ - vị ngữ - giá trị, dùng đúng "
        "từ ngữ trong đoạn trích. Không thêm giải thích, không thêm câu thứ hai.\n\n"
        "Ví dụ - Đoạn trích: \"[1] Tháp Eiffel - Wikipedia: Tháp Eiffel là công trình bằng "
        "thép cao 330 mét kể cả ăng-ten, hoàn thành năm 1889.\"\n"
        "Yêu cầu: Tháp Eiffel cao bao nhiêu mét?\n"
        "Trả lời: Tháp Eiffel cao 330 mét."
    ),
    # Khảo sát v1: chỉ mô tả quy tắc trừu tượng ("nếu multi-hop thì tách"), bắt
    # model tự suy luận khi nào áp dụng - kết quả rất tệ (xem README): bỏ sót
    # đúng case multi-hop cần tách nhất, sinh placeholder degenerate
    # ("câu hỏi con 1"), hoặc tách mất thuộc tính (mất "chiều cao" khi tách câu
    # so sánh 2 tháp). Model cỡ 0.5B không có khả năng suy luận điều kiện ổn
    # định - đừng bắt nó suy luận "khi nào nên tách", đưa thẳng ví dụ cụ thể để
    # nó bắt chước pattern (few-shot) thay vì diễn giải quy tắc trừu tượng hơn.
    "decompose": (
        "Tách câu hỏi thành các câu hỏi con để search, theo đúng 3 ví dụ dưới đây. "
        "Dùng {X} để chỉ kết quả câu hỏi con ngay trước, nếu câu hỏi con sau cần dùng nó.\n\n"
        'Câu hỏi: "Thủ đô của nước giáp Việt Nam ở phía Tây là gì?"\n'
        'Trả lời: {"steps": ["Nước nào giáp Việt Nam ở phía Tây?", "Thủ đô của {X} là gì?"]}\n\n'
        'Câu hỏi: "Tháp Eiffel cao hơn tháp Big Ben bao nhiêu mét?"\n'
        'Trả lời: {"steps": ["Tháp Eiffel cao bao nhiêu mét?", "Tháp Big Ben cao bao nhiêu mét?"]}\n\n'
        'Câu hỏi: "Thủ đô Nhật Bản là gì?"\n'
        'Trả lời: {"steps": ["Thủ đô Nhật Bản là gì?"]}\n\n'
        "Chỉ trả về JSON đúng định dạng như trên, không giải thích, không thêm chữ nào khác."
    ),
    # Khảo sát v2: thay vì bắt model tự tạo ra câu hỏi con (sinh chữ mới, dễ
    # hallucinate - xem case "phía Tây" ở mode decompose), chỉ bắt model GẮN
    # NHÃN các cụm có sẵn trong câu hỏi gốc (giống việc trích nguyên văn ở mode
    # extract, vốn đáng tin hơn synth). Python sẽ tự ráp câu hỏi con từ các
    # nhãn này (regex._needs_lookup vẫn giữ làm lưới an toàn trước khi quyết
    # định có tách hay không) - model chỉ làm việc "định vị cụm từ", không tự
    # quyết "có nên tách" hay "tách thế nào".
    "tag": (
        "Phân tích cấu trúc câu hỏi theo 4 ví dụ dưới đây. Xác định 4 phần, LẤY NGUYÊN "
        "VĂN từ câu hỏi gốc, không viết lại: chu_ngu (cụm danh từ chính đang được hỏi), "
        "tu_noi (từ nối như \"của\"/\"hơn\"/\"so với\", để rỗng \"\" nếu câu không có từ nối "
        "rõ ràng), phu_ngu (cụm có thể là một thực thể riêng cần tìm hiểu thêm, để rỗng \"\" "
        "nếu không có), phan_hoi (phần hỏi ở cuối câu).\n\n"
        'Câu hỏi: "Thủ đô của nước láng giềng phía bắc Việt Nam là gì?"\n'
        'Trả lời: {"chu_ngu": "Thủ đô", "tu_noi": "của", "phu_ngu": "nước láng giềng phía bắc '
        'Việt Nam", "phan_hoi": "là gì"}\n\n'
        'Câu hỏi: "Dân số thành phố lớn nhất Nhật Bản là bao nhiêu?"\n'
        'Trả lời: {"chu_ngu": "Dân số", "tu_noi": "", "phu_ngu": "thành phố lớn nhất Nhật Bản", '
        '"phan_hoi": "là bao nhiêu"}\n\n'
        'Câu hỏi: "Tháp Eiffel cao hơn tháp Tokyo Skytree bao nhiêu mét?"\n'
        'Trả lời: {"chu_ngu": "Tháp Eiffel cao", "tu_noi": "hơn", "phu_ngu": "tháp Tokyo '
        'Skytree", "phan_hoi": "bao nhiêu mét"}\n\n'
        'Câu hỏi: "Thủ đô Việt Nam là gì?"\n'
        'Trả lời: {"chu_ngu": "Thủ đô Việt Nam", "tu_noi": "", "phu_ngu": "", "phan_hoi": "là gì"}\n\n'
        "Chỉ trả về JSON đúng định dạng như trên, không giải thích, không thêm chữ nào khác."
    ),
    # Dùng cho adaptive_entity.py (bước bridge trong chain.py) - khác
    # extract_unconditional ở chỗ hẹp hơn nhiều: chỉ cần 1 tên riêng, không
    # phải 1 câu/cụm bất kỳ. extract_unconditional vẫn để model tự do trích
    # "câu/cụm phù hợp" nên hay lan man vào việc nhắc lại tiêu đề/phần dẫn của
    # snippet trước khi vào nội dung ("[1] Bài 19: Các nước láng...") - cắt
    # ngắn n_predict không sửa được việc này (chỉ cắt cụt giữa chừng phần lan
    # man, xem README), phải chặn ngay từ prompt.
    "extract_entity": (
        "Bạn là một công cụ tìm kiếm, không phải trợ lý hội thoại. Đọc đoạn trích dưới đây "
        "và trả lời câu hỏi bằng ĐÚNG MỘT tên riêng (tên quốc gia, thành phố, người, tổ chức...). "
        "KHÔNG lặp lại tiêu đề đoạn trích, KHÔNG kèm số thứ tự kiểu \"[1]\", KHÔNG giải thích, "
        "KHÔNG viết thành câu. Chỉ in ra đúng cái tên đó, không gì khác."
    ),
    # 3 prompt cho adaptive_qwen3.py - chain 3 bước độc lập tận dụng reasoning
    # của Qwen3 (không tắt bằng /no_think như các nơi khác dùng Qwen3), thay
    # vì 1 lần gọi extract_unconditional như adaptive.py gốc. Mỗi bước là 1
    # request HTTP riêng (không phải multi-turn conversation) - "độc lập" theo
    # đúng nghĩa: bước sau không thấy được quá trình suy luận (reasoning_content)
    # của bước trước, chỉ thấy answer cuối (content) của nó, giống cách 1
    # reviewer con người không cần đọc hết bản nháp, chỉ cần đọc bản đã chốt.
    "qwen3_extract": (
        "Đọc đoạn trích dưới đây và câu hỏi. Suy luận từng bước xem đoạn trích có chứa "
        "thông tin trả lời trực tiếp câu hỏi hay không. Nếu có, trích dẫn NGUYÊN VĂN câu/cụm "
        "chứa thông tin đó rồi nêu câu trả lời ngắn gọn. Nếu không có, nói rõ \"không có thông tin\"."
    ),
    # Ban dau viet "suy dien" la 1 tieu chi bac bo, khong phan biet suy luan
    # hop ly TU thong tin co that trong doan trich (vd doan trich noi "Trung
    # Quoc giap VN o phia Bac" + "thu do Trung Quoc la Bac Kinh" -> suy ra
    # "Bac Kinh" la HOP LE) voi bia them noi dung KHONG he co trong doan trich
    # - hau qua thay truc tiep: critique bac bo dung cau tra loi dung ("Bac
    # Kinh") vi cho la "suy dien", trong khi no duoc suy tu chinh noi dung
    # doan trich. Sua lai de phan biet ro 2 truong hop.
    "qwen3_critique": (
        "Bạn là người kiểm tra chất lượng độc lập, không phải người đã trả lời câu hỏi. Cho "
        "đoạn trích gốc, câu hỏi, và một câu trả lời ứng viên - suy luận từng bước xem MỌI thông "
        "tin trong câu trả lời có thực sự xuất hiện hoặc suy ra được trực tiếp từ nội dung đoạn "
        "trích hay không. Suy luận hợp lý dựa trên thông tin CÓ THẬT trong đoạn trích là HỢP LỆ "
        "(ví dụ: đoạn trích nói \"Trung Quốc giáp Việt Nam ở phía Bắc\" và \"thủ đô Trung Quốc là "
        "Bắc Kinh\" thì suy ra \"Bắc Kinh\" là HỢP LỆ, dù đoạn trích không nói thẳng \"Bắc Kinh\" "
        "là câu trả lời) - khác với bịa thêm thông tin KHÔNG hề xuất hiện trong đoạn trích (KHÔNG "
        "HỢP LỆ). Kết luận CHỈ MỘT trong hai, ở dòng cuối cùng: \"HOP_LE\" nếu mọi thông tin trong "
        "câu trả lời đều có căn cứ (trực tiếp hoặc suy luận hợp lý) trong đoạn trích, hoặc "
        "\"KHONG_HOP_LE\" nếu câu trả lời sai hoặc đưa vào thông tin không hề xuất hiện trong "
        "đoạn trích."
    ),
    "qwen3_finalize": (
        "Cho câu hỏi, một câu trả lời ứng viên, và kết quả kiểm tra (HOP_LE hoặc KHONG_HOP_LE). "
        "Nếu kiểm tra là HOP_LE, in ra câu trả lời ứng viên (có thể viết gọn hơn nếu dài dòng, "
        "giữ nguyên nội dung). Nếu kiểm tra là KHONG_HOP_LE, in ra đúng câu: \"không có thông "
        "tin\". Không giải thích gì thêm, không nhắc lại kết quả kiểm tra."
    ),
}


def _server_ready() -> bool:
    # /health trả 503 trong lúc model đang load, 200 khi model sẵn sàng nhận request.
    try:
        return requests.get(f"{SERVER_URL}/health", timeout=1).status_code == 200
    except requests.RequestException:
        return False


def _loaded_model_path() -> str | None:
    try:
        data = requests.get(f"{SERVER_URL}/v1/models", timeout=2).json()
        return data["data"][0]["id"]
    except (requests.RequestException, KeyError, IndexError):
        return None


def _llama_server_pids() -> list[str]:
    # `pkill`/`pgrep -x llama-server` KHÔNG khớp được process trên máy này dù
    # `ps aux` thấy rõ nó (đã kiểm chứng trực tiếp: pkill trả exit code 1 =
    # "không có process khớp" trong khi ps aux vẫn liệt kê nó) - lý do khiến
    # _stop_server() từng âm thầm thất bại suốt (server cũ không hề bị dừng,
    # ensure_server() tưởng nhầm nó vẫn dùng được). Tự parse `ps aux` lấy PID
    # thay vì dựa vào khớp tên qua pkill/pgrep.
    out = subprocess.run(["ps", "aux"], capture_output=True, text=True, check=False).stdout
    return [
        parts[1] for line in out.splitlines()
        if len(parts := line.split()) > 10 and parts[10].endswith("/llama-server")
    ]


def _running_server_bin() -> str | None:
    # Đọc /proc/<pid>/exe để biết server đang chạy dùng đúng binary nào (pkg
    # hay tự build) - _loaded_model_path() chỉ biết model, không phân biệt
    # được 2 binary khác nhau load cùng 1 file model.
    pids = _llama_server_pids()
    try:
        return os.readlink(f"/proc/{pids[0]}/exe") if pids else None
    except OSError:
        return None


def _stop_server() -> None:
    for pid in _llama_server_pids():
        subprocess.run(["kill", "-9", pid], check=False)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and _server_ready():
        time.sleep(0.3)


_ensure_server_lock = threading.Lock()


def ensure_server(startup_timeout: float = 60) -> None:
    # adaptive.py gọi run_model() (=> ensure_server()) từ nhiều thread song
    # song - không khóa thì nhiều thread cùng thấy server chưa sẵn sàng và
    # cùng tự Popen llama-server, 2 process giành 1 port (đã thấy log lẫn
    # "couldn't bind" + model load chồng nhau, 1 request bị trễ 18s vô lý).
    # Khóa cả hàm: sau khi server đã warm, mỗi lần gọi chỉ là 1 GET /health
    # rẻ, không đáng để tối ưu ra khỏi lock.
    with _ensure_server_lock:
        # Android hạ xung cụm core "big" cho app chạy nền (Termux không phải
        # app foreground) - đã đo được gen_tps rơi từ ~27 tok/s xuống
        # 0.17 tok/s (thấy nhầm là do RAM). termux-wake-lock giữ CPU không bị
        # Doze/background throttle. Gọi lại vẫn an toàn nếu đã lock rồi.
        subprocess.run(["termux-wake-lock"], check=False)

        if _server_ready():
            # Server đang chạy có thể là của một lần gọi trước với model khác
            # hoặc binary khác (vd so sánh pkg vs build từ source) - phải
            # kiểm tra khớp cả MODEL_PATH lẫn LLAMA_SERVER_BIN, không thì âm
            # thầm tái sử dụng nhầm process cũ (đã thấy bug này: benchmark
            # "pkg" nhưng thực ra vẫn đang đo binary build từ source).
            if _loaded_model_path() == str(MODEL_PATH) and _running_server_bin() == str(LLAMA_SERVER_BIN):
                return
            _stop_server()

        log = SERVER_LOG.open("w")
        subprocess.Popen(
            [
                str(LLAMA_SERVER_BIN), "-m", str(MODEL_PATH), "--host", "127.0.0.1", "--port", "8080",
                "-np", str(N_PARALLEL), "-c", str(CTX_SIZE),
            ],
            stdout=log, stderr=log, start_new_session=True,
        )
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if _server_ready():
                return
            time.sleep(0.5)
        raise RuntimeError(f"llama-server không sẵn sàng sau {startup_timeout}s, xem {SERVER_LOG}")


def build_user_prompt(question: str, snippets: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i + 1}] {s['title']}: {s['snippet']}" for i, s in enumerate(snippets)
    )
    return f"Đoạn trích:\n{context}\n\nCâu hỏi: {question}"


def run_model(
    user_prompt: str, mode: str = "synth", n_predict: int = 200,
    temperature: float | None = None, timeout: float = 300,
) -> dict:
    ensure_server()
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[mode]},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": n_predict,
    }
    # None -> giữ nguyên default của llama-server (dùng cho pipeline một-lượt
    # hiện có). Self-consistency (adaptive.py) cần set > 0 để có đa dạng giữa
    # các lần sinh, nếu không voting vô nghĩa vì mọi lần sinh sẽ giống hệt nhau.
    if temperature is not None:
        payload["temperature"] = temperature
    resp = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    timings = data.get("timings", {})
    choice = data["choices"][0]
    message = choice["message"]
    # finish_reason/completion_tokens/reasoning_content: dùng để debug các
    # chain nhiều bước với model reasoning (Qwen3, xem adaptive_qwen3.py) -
    # finish_reason="length" nghĩa là model bị cắt cụt bởi n_predict trước
    # khi tự dừng (có thể vẫn đang ở giữa <think>, content rỗng dù model
    # "làm việc" thật) khác với "stop" (tự kết luận xong). reasoning_content
    # chỉ có ở model dual-mode reasoning không dùng /no_think - completion_tokens
    # tính GỘP cả reasoning_content lẫn content, không tách riêng được từ API.
    return {
        "answer": message["content"].strip(),
        "reasoning": message.get("reasoning_content"),
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": data.get("usage", {}).get("completion_tokens"),
        "prompt_tps": timings.get("prompt_per_second"),
        "gen_tps": timings.get("predicted_per_second"),
    }


def answer_question(question: str, n_results: int = 5, mode: str = "synth") -> dict:
    snippets = search(question, n=n_results)
    user_prompt = build_user_prompt(question, snippets)
    model_result = run_model(user_prompt, mode=mode)
    return {"question": question, "snippets": snippets, **model_result}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Thủ đô Việt Nam là gì?"
    out = answer_question(q)
    print(out["answer"])
    print(f"\n[prompt_tps={out['prompt_tps']} gen_tps={out['gen_tps']}]")
