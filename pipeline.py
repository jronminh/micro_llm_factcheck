"""Search + model, ghép prompt ép model chỉ tổng hợp trên snippet đã search được.

Model chạy qua llama-server (HTTP, OpenAI-compatible). Lý do không dùng
llama-cli trực tiếp: ở chat/conversation mode, llama-cli echo lại prompt vào
stdout xen với câu trả lời (và cắt ngắn phần echo khi prompt dài), không có
ranh giới rõ ràng để tách câu trả lời thật ra bằng regex. llama-server trả
JSON sạch, đồng thời giữ model warm giữa các lần gọi.
"""
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


def _stop_server() -> None:
    # -x (khớp đúng tên process) thay vì -f: -f khớp cả cmdline của process
    # gọi pkill, có thể tự kill nhầm shell đang chạy nó.
    subprocess.run(["pkill", "-x", "llama-server"], check=False)
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
            # - phải kiểm tra khớp MODEL_PATH, không thì âm thầm benchmark
            # nhầm model.
            if _loaded_model_path() == str(MODEL_PATH):
                return
            _stop_server()

        log = SERVER_LOG.open("w")
        subprocess.Popen(
            [
                "llama-server", "-m", str(MODEL_PATH), "--host", "127.0.0.1", "--port", "8080",
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
    return {
        "answer": data["choices"][0]["message"]["content"].strip(),
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
