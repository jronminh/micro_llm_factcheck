"""Trích xuất độc lập trên từng link rồi lấy đồng thuận, thay vì một lần gọi
cố định như pipeline.answer_question(). Khác với self-consistency (sampling
lặp lại trên cùng một context gộp), đa dạng ở đây đến từ nguồn khác nhau: nếu
nhiều link độc lập cùng cho ra một câu trả lời, đó là tín hiệu đối chiếu chéo
thật, không phải nhiễu ngẫu nhiên của model.

Xử lý theo batch N_PARALLEL link cùng lúc (khớp số slot của llama-server),
kiểm tra đồng thuận sau mỗi batch (early-stop khi đạt ngưỡng) để không lãng
phí lượt gọi model khi đã đủ tin cậy sớm. Trong 1 batch không có early-stop
giữa batch - đã gửi hết batch thì đợi hết batch mới check, đổi lại tốc độ
~N_PARALLEL lần so với tuần tự khi cần đi hết hoặc gần hết max_n link.

Các tham số (max_n, consensus_threshold, min_votes, timeout_s) chưa có con số
hợp lý - đang trong giai đoạn tune bằng benchmark thực tế.

n_predict=80 (thay vì default 200 của run_model): mode extract chỉ cần copy
một câu/cụm ngắn, không cần sinh dài. Quan sát qua benchmark: các câu trả
lời tệ nhất (rambling, tính toán sai) đều là lúc model sinh dài - cắt ngắn
n_predict vừa nhanh hơn (bottleneck là gen_tps, không phải prompt_tps) vừa
ép model dừng trước khi rambling ra khỏi phạm vi trích xuất.
"""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlparse

import requests

import pipeline
from pipeline import N_PARALLEL, build_user_prompt, run_model
from score import score_answer
from search import search

# Cache (câu hỏi, link) -> kết quả trích xuất, sống qua các lần chạy CLI khác
# nhau - khi tune consensus_threshold/min_votes trên cùng một câu hỏi, không
# cần gọi lại model cho cặp đã xử lý. Không commit vào git (xem .gitignore),
# chỉ để tune cục bộ.
CACHE_PATH = Path(__file__).parent / ".link_cache.json"


def _cache_key(question: str, snippet: dict, n_predict: int, temperature: float) -> str:
    # n_predict/temperature ảnh hưởng trực tiếp đến answer sinh ra - phải nằm
    # trong key, nếu không đổi tham số này sẽ âm thầm trả về answer cache từ
    # lần tune trước với n_predict/temperature khác.
    #
    # MODEL_PATH cũng phải nằm trong key - phát hiện được khi so sánh
    # Qwen3-0.6B với Qwen2.5-0.5B (explore_qwen3.py): thiếu MODEL_PATH khiến
    # 4/8 câu hỏi âm thầm trả về answer cache từ lần chạy Qwen2.5 trước đó
    # thay vì thực sự gọi Qwen3 - kết quả so sánh sai mà không có dấu hiệu
    # lỗi nào (answer vẫn "hợp lệ", chỉ là của model khác).
    raw = (
        f"{question}||{n_predict}||{temperature}||{pipeline.MODEL_PATH}||"
        f"{snippet.get('url') or snippet.get('snippet', '')}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _cluster(texts: list[str], sim_threshold: float) -> list[list[int]]:
    """Gom chỉ số các câu trả lời giống nhau (SequenceMatcher >= sim_threshold).

    Greedy, không phải tối ưu toàn cục - đủ dùng vì số link nhỏ (<=15).
    """
    clusters: list[list[int]] = []
    reps: list[str] = []
    for i, text in enumerate(texts):
        for cluster, rep in zip(clusters, reps):
            if SequenceMatcher(None, rep.lower().strip(), text.lower().strip()).ratio() >= sim_threshold:
                cluster.append(i)
                break
        else:
            clusters.append([i])
            reps.append(text)
    return clusters


def _content_bias(snippets: list[dict], content_sim_threshold: float) -> float | None:
    """Bias theo NỘI DUNG snippet gốc, không phải domain - bắt được trường hợp
    nhiều domain khác nhau nhưng cùng chép/diễn giải một nguồn gốc (bias theo
    domain không thấy được, xem README case "thủ đô nước láng giềng phía bắc
    VN": 10 domain khác nhau nhưng cùng paraphrase 1 bài SGK).

    Tái dùng _cluster() như với answers, nhưng input là title+snippet gốc.
    Trả về tỉ lệ cụm nội dung lớn nhất / tổng số snippet trong cụm thắng -
    cùng thang đo với bias theo domain (0 = mọi snippet khác nhau thật, gần 1
    = phần lớn snippet trong cụm thắng là bản chép/diễn giải của nhau).
    """
    if len(snippets) <= 1:
        return 0.0
    texts = [f"{s.get('title', '')} {s.get('snippet', '')}".strip() for s in snippets]
    clusters = _cluster(texts, content_sim_threshold)
    largest = max(len(c) for c in clusters)
    return round(largest / len(texts), 2)


def _best_cluster_result(
    question: str, answers: list[tuple[str, str]], snippets: list[dict],
    cluster_sim_threshold: float, content_sim_threshold: float,
) -> dict | None:
    # Mẫu degenerate ("[2]", rỗng...) không tham gia cluster - nếu không,
    # nhiều lần degenerate giống hệt nhau có thể thắng đồng thuận dù không có
    # nội dung thật. snippets phải lọc song song với answers để giữ đúng chỉ
    # số khi tính domain của cluster thắng (bias).
    usable = [(a, v, s) for (a, v), s in zip(answers, snippets) if v != "degenerate"]
    if not usable:
        return None
    texts = [a for a, _, _ in usable]
    clusters = _cluster(texts, cluster_sim_threshold)
    clusters.sort(key=len, reverse=True)
    top = clusters[0]
    representative = max((usable[i][0] for i in top), key=len)
    # bias: tỉ lệ trùng domain trong cụm thắng - 0 nghĩa là mọi link đến từ
    # domain khác nhau (đồng thuận độc lập thật), gần 1 nghĩa là cụm thắng
    # chủ yếu dựa vào rất ít domain lặp lại (rủi ro echo-chamber, nhiều trang
    # cùng chia sẻ một nội dung/lỗi gốc - xem case "thủ đô nước láng giềng
    # phía bắc VN" trong README, agreement cao nhưng sai vì corpus lệch).
    domains = [urlparse(usable[i][2].get("url") or "").hostname for i in top]
    domains = [d for d in domains if d]
    bias = round(1 - len(set(domains)) / len(domains), 2) if domains else None
    content_bias = _content_bias([usable[i][2] for i in top], content_sim_threshold)
    # Chia theo tổng số link đã thử (kể cả degenerate), để link hỏng vẫn kéo
    # đồng thuận xuống thay vì bị coi như chưa từng xảy ra.
    return {
        "answer": representative, "agreement": len(top) / len(answers), "cluster_size": len(top),
        "bias": bias, "content_bias": content_bias,
    }


def _process_link(
    question: str, snippet: dict, n_predict: int, temperature: float, per_link_timeout_s: float,
) -> tuple[str, str, float] | None:
    """Gọi model cho 1 link, chạy trong thread riêng - trả None nếu timeout/lỗi.

    Không đụng vào `cache` dict (đọc/ghi cache chỉ làm ở main thread, sau khi
    join batch) để tránh race condition khi nhiều thread chạy đồng thời.
    """
    user_prompt = build_user_prompt(question, [snippet])
    t0 = time.monotonic()
    try:
        result = run_model(
            user_prompt, mode="extract_unconditional", temperature=temperature,
            n_predict=n_predict, timeout=per_link_timeout_s,
        )
    except requests.exceptions.RequestException:
        return None
    elapsed = time.monotonic() - t0
    answer = result["answer"]
    verdict = score_answer(question, answer, [snippet])["verdict"]
    return answer, verdict, elapsed


def answer_question_per_link(
    question: str,
    max_n: int = 8,
    consensus_threshold: float = 0.75,
    cluster_sim_threshold: float = 0.7,
    content_sim_threshold: float = 0.5,
    min_votes: int = 4,
    temperature: float = 0.0,
    n_predict: int = 80,
    per_link_timeout_s: float = 20,
    timeout_s: float = 90,
    use_cache: bool = True,
    debug: bool = False,
) -> dict:
    deadline = time.monotonic() + timeout_s
    snippets = search(question, n=max_n)
    if debug:
        print(f"[debug] search: {len(snippets)} link cho {question!r}", flush=True)

    cache = _load_cache() if use_cache else {}
    answers: list[tuple[str, str]] = []
    used_snippets: list[dict] = []  # song song với answers - chỉ chứa link thật sự có kết quả
    n_attempted = 0  # kể cả link timeout/lỗi - answers/used_snippets chỉ chứa link có phản hồi

    for batch_start in range(0, len(snippets), N_PARALLEL):
        if time.monotonic() >= deadline:
            if debug:
                print(f"[debug] hết timeout_s={timeout_s}s trước link {batch_start + 1}, dừng", flush=True)
            break
        batch = list(enumerate(snippets[batch_start:batch_start + N_PARALLEL], start=batch_start + 1))
        n_attempted += len(batch)

        to_run = []  # (idx, snippet, key) - phần không có cache, phải gọi model
        for idx, snippet in batch:
            key = _cache_key(question, snippet, n_predict, temperature)
            cached = cache.get(key) if use_cache else None
            if cached:
                answer, verdict = cached["answer"], cached["verdict"]
                answers.append((answer, verdict))
                used_snippets.append(snippet)
                if debug:
                    print(f"[debug] link {idx}/{len(snippets)} (cache) verdict={verdict}: {answer[:80]!r}", flush=True)
            else:
                to_run.append((idx, snippet, key))

        if to_run:
            with ThreadPoolExecutor(max_workers=len(to_run)) as executor:
                futures = {
                    executor.submit(_process_link, question, snippet, n_predict, temperature, per_link_timeout_s):
                        (idx, snippet, key)
                    for idx, snippet, key in to_run
                }
                for future in as_completed(futures):
                    idx, snippet, key = futures[future]
                    result = future.result()
                    if result is None:
                        if debug:
                            print(f"[debug] link {idx}/{len(snippets)} bỏ qua (timeout/lỗi)", flush=True)
                        continue
                    answer, verdict, elapsed = result
                    if use_cache:
                        cache[key] = {"answer": answer, "verdict": verdict}
                    answers.append((answer, verdict))
                    used_snippets.append(snippet)
                    if debug:
                        preview = answer[:80].replace("\n", " ")
                        print(
                            f"[debug] link {idx}/{len(snippets)} ({elapsed:.1f}s) verdict={verdict} "
                            f"src={snippet.get('title', '')[:50]!r}\n         -> {preview!r}",
                            flush=True,
                        )
            if use_cache:
                # Ghi 1 lần sau khi cả batch join xong (main thread, không có
                # thread nào khác đụng vào `cache` dict) - tránh race condition
                # và giảm I/O so với ghi mỗi link.
                _save_cache(cache)

        if len(answers) < min_votes:
            continue
        best = _best_cluster_result(question, answers, used_snippets, cluster_sim_threshold, content_sim_threshold)
        if debug and best:
            print(
                f"[debug]   đồng thuận: {best['agreement']:.2f} (ngưỡng {consensus_threshold}) "
                f"coverage={best['cluster_size'] / n_attempted:.2f} bias={best['bias']} "
                f"content_bias={best['content_bias']}",
                flush=True,
            )
        if best and best["agreement"] >= consensus_threshold:
            if debug:
                print(f"[debug] đạt ngưỡng sau {len(answers)} link, dừng sớm", flush=True)
            return {
                "question": question, "answer": best["answer"],
                "verdict": score_answer(question, best["answer"], used_snippets)["verdict"],
                "n_links_used": len(answers), "agreement": round(best["agreement"], 2),
                "coverage": round(best["cluster_size"] / n_attempted, 2), "bias": best["bias"],
                "content_bias": best["content_bias"],
                "confident": True, "snippets": used_snippets,
            }

    # Hết link hoặc hết giờ mà chưa đạt ngưỡng - trả cụm lớn nhất đã có.
    best = _best_cluster_result(question, answers, used_snippets, cluster_sim_threshold, content_sim_threshold)
    if debug:
        print(
            f"[debug] hết link/timeout sau {len(answers)} link, "
            f"agreement={best['agreement']:.2f}" if best else "[debug] không có mẫu nào dùng được",
            flush=True,
        )
    if best is None:
        return {"question": question, "answer": None, "verdict": "no_sample", "confident": False}
    return {
        "question": question, "answer": best["answer"],
        "verdict": score_answer(question, best["answer"], used_snippets)["verdict"],
        "n_links_used": len(answers), "agreement": round(best["agreement"], 2),
        "coverage": round(best["cluster_size"] / n_attempted, 2), "bias": best["bias"],
        "content_bias": best["content_bias"],
        "confident": False, "snippets": used_snippets,
    }


if __name__ == "__main__":
    import sys

    import pipeline

    # Model nhỏ hơn để lặp tham số nhanh hơn khi chưa có con số hợp lý.
    pipeline.MODEL_PATH = Path(__file__).parent / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

    q = " ".join(sys.argv[1:]) or "Thủ đô Việt Nam là gì?"
    t0 = time.monotonic()
    out = answer_question_per_link(q, debug=True)
    elapsed = time.monotonic() - t0

    print(f"Câu hỏi: {q}")
    print(f"Trả lời: {out['answer']}")
    print(
        f"verdict={out['verdict']} confident={out.get('confident')} "
        f"n_links_used={out.get('n_links_used')} agreement={out.get('agreement')} "
        f"coverage={out.get('coverage')} bias={out.get('bias')} content_bias={out.get('content_bias')} "
        f"elapsed={elapsed:.1f}s"
    )
