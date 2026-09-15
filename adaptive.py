"""Trích xuất độc lập trên từng link rồi lấy đồng thuận, thay vì một lần gọi
cố định như pipeline.answer_question(). Khác với self-consistency (sampling
lặp lại trên cùng một context gộp), đa dạng ở đây đến từ nguồn khác nhau: nếu
nhiều link độc lập cùng cho ra một câu trả lời, đó là tín hiệu đối chiếu chéo
thật, không phải nhiễu ngẫu nhiên của model.

Xử lý tuần tự từng link, kiểm tra đồng thuận sau mỗi link (early-stop khi đạt
ngưỡng) để không lãng phí lượt gọi model khi đã đủ tin cậy sớm.

Các tham số (max_n, consensus_threshold, min_votes, timeout_s) chưa có con số
hợp lý - đang trong giai đoạn tune bằng benchmark thực tế.
"""
import hashlib
import json
import time
from difflib import SequenceMatcher
from pathlib import Path

import requests

from pipeline import build_user_prompt, run_model
from score import score_answer
from search import search

# Cache (câu hỏi, link) -> kết quả trích xuất, sống qua các lần chạy CLI khác
# nhau - khi tune consensus_threshold/min_votes trên cùng một câu hỏi, không
# cần gọi lại model cho cặp đã xử lý. Không commit vào git (xem .gitignore),
# chỉ để tune cục bộ.
CACHE_PATH = Path(__file__).parent / ".link_cache.json"


def _cache_key(question: str, snippet: dict) -> str:
    raw = question + "||" + (snippet.get("url") or snippet.get("snippet", ""))
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


def _best_cluster_result(question: str, answers: list[tuple[str, str]], cluster_sim_threshold: float) -> dict | None:
    # Mẫu degenerate ("[2]", rỗng...) không tham gia cluster - nếu không,
    # nhiều lần degenerate giống hệt nhau có thể thắng đồng thuận dù không có
    # nội dung thật.
    usable = [(a, v) for a, v in answers if v != "degenerate"]
    if not usable:
        return None
    texts = [a for a, _ in usable]
    clusters = _cluster(texts, cluster_sim_threshold)
    clusters.sort(key=len, reverse=True)
    top = clusters[0]
    representative = max((usable[i][0] for i in top), key=len)
    # Chia theo tổng số link đã thử (kể cả degenerate), để link hỏng vẫn kéo
    # đồng thuận xuống thay vì bị coi như chưa từng xảy ra.
    return {"answer": representative, "agreement": len(top) / len(answers)}


def answer_question_per_link(
    question: str,
    max_n: int = 8,
    consensus_threshold: float = 0.6,
    cluster_sim_threshold: float = 0.7,
    min_votes: int = 3,
    temperature: float = 0.0,
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

    for idx, snippet in enumerate(snippets, start=1):
        if time.monotonic() >= deadline:
            if debug:
                print(f"[debug] hết timeout_s={timeout_s}s trước link {idx}, dừng", flush=True)
            break

        key = _cache_key(question, snippet)
        cached = cache.get(key) if use_cache else None
        if cached:
            answer, verdict = cached["answer"], cached["verdict"]
            if debug:
                print(f"[debug] link {idx}/{len(snippets)} (cache) verdict={verdict}: {answer[:80]!r}", flush=True)
        else:
            user_prompt = build_user_prompt(question, [snippet])
            t0 = time.monotonic()
            try:
                # Timeout riêng từng link - adaptive nghĩa là còn nhiều link
                # khác để thử, không đáng đợi một link chậm bất thường (đã
                # thấy outlier 70s+ do RAM pressure trên máy này).
                result = run_model(
                    user_prompt, mode="extract_unconditional", temperature=temperature,
                    timeout=per_link_timeout_s,
                )
            except requests.exceptions.RequestException as exc:
                if debug:
                    print(f"[debug] link {idx}/{len(snippets)} bỏ qua (timeout/lỗi: {exc})", flush=True)
                continue
            elapsed = time.monotonic() - t0
            answer = result["answer"]
            verdict = score_answer(question, answer, [snippet])["verdict"]
            if use_cache:
                cache[key] = {"answer": answer, "verdict": verdict}
                _save_cache(cache)
            if debug:
                preview = answer[:80].replace("\n", " ")
                print(
                    f"[debug] link {idx}/{len(snippets)} ({elapsed:.1f}s) verdict={verdict} "
                    f"src={snippet.get('title', '')[:50]!r}\n         -> {preview!r}",
                    flush=True,
                )

        answers.append((answer, verdict))
        used_snippets.append(snippet)

        if len(answers) < min_votes:
            continue
        best = _best_cluster_result(question, answers, cluster_sim_threshold)
        if debug and best:
            print(f"[debug]   đồng thuận: {best['agreement']:.2f} (ngưỡng {consensus_threshold})", flush=True)
        if best and best["agreement"] >= consensus_threshold:
            if debug:
                print(f"[debug] đạt ngưỡng sau {len(answers)} link, dừng sớm", flush=True)
            return {
                "question": question, "answer": best["answer"],
                "verdict": score_answer(question, best["answer"], used_snippets)["verdict"],
                "n_links_used": len(answers), "agreement": round(best["agreement"], 2),
                "confident": True, "snippets": used_snippets,
            }

    # Hết link hoặc hết giờ mà chưa đạt ngưỡng - trả cụm lớn nhất đã có.
    best = _best_cluster_result(question, answers, cluster_sim_threshold)
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
        f"elapsed={elapsed:.1f}s"
    )
