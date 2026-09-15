"""Search + model, ghép prompt ép model chỉ tổng hợp trên snippet đã search được.

Model chạy qua llama-server (HTTP, OpenAI-compatible). Lý do không dùng
llama-cli trực tiếp: ở chat/conversation mode, llama-cli echo lại prompt vào
stdout xen với câu trả lời (và cắt ngắn phần echo khi prompt dài), không có
ranh giới rõ ràng để tách câu trả lời thật ra bằng regex. llama-server trả
JSON sạch, đồng thời giữ model warm giữa các lần gọi.
"""
import subprocess
import sys
import time
from pathlib import Path

import requests

from search import search

MODEL_PATH = Path(__file__).parent / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
SERVER_URL = "http://127.0.0.1:8080"
SERVER_LOG = Path(__file__).parent / ".server.log"

SYSTEM_PROMPT = (
    "Bạn chỉ được trả lời dựa trên các đoạn trích dưới đây, không dùng kiến thức "
    "nào khác. Khi trả lời, trích dẫn đúng câu chữ trong đoạn trích liên quan. "
    "Nếu các đoạn trích không đủ thông tin để trả lời, nói rõ \"không có thông tin\"."
)


def ensure_server(startup_timeout: float = 60) -> None:
    try:
        requests.get(f"{SERVER_URL}/health", timeout=1)
        return
    except requests.RequestException:
        pass

    log = SERVER_LOG.open("w")
    subprocess.Popen(
        ["llama-server", "-m", str(MODEL_PATH), "--host", "127.0.0.1", "--port", "8080"],
        stdout=log, stderr=log, start_new_session=True,
    )
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        try:
            requests.get(f"{SERVER_URL}/health", timeout=1)
            return
        except requests.RequestException:
            time.sleep(0.5)
    raise RuntimeError(f"llama-server không sẵn sàng sau {startup_timeout}s, xem {SERVER_LOG}")


def build_user_prompt(question: str, snippets: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i + 1}] {s['title']}: {s['snippet']}" for i, s in enumerate(snippets)
    )
    return f"Đoạn trích:\n{context}\n\nCâu hỏi: {question}"


def run_model(user_prompt: str, n_predict: int = 200) -> dict:
    ensure_server()
    resp = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": n_predict,
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    timings = data.get("timings", {})
    return {
        "answer": data["choices"][0]["message"]["content"].strip(),
        "prompt_tps": timings.get("prompt_per_second"),
        "gen_tps": timings.get("predicted_per_second"),
    }


def answer_question(question: str, n_results: int = 5) -> dict:
    snippets = search(question, n=n_results)
    user_prompt = build_user_prompt(question, snippets)
    model_result = run_model(user_prompt)
    return {"question": question, "snippets": snippets, **model_result}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Thủ đô Việt Nam là gì?"
    out = answer_question(q)
    print(out["answer"])
    print(f"\n[prompt_tps={out['prompt_tps']} gen_tps={out['gen_tps']}]")
