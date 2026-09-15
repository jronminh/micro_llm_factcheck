"""Fork của adaptive.py, chuyên cho bước bridge trong chain.py: tìm 1 tên riêng
ngắn (tên nước/thành phố/người) thay vì 1 câu trả lời fact-check đầy đủ.

Lý do fork thay vì thêm tham số mode vào adaptive.py: yêu cầu ở đây hẹp hơn
hẳn (đúng 1 tên riêng, không phải câu/cụm bất kỳ) - dùng prompt riêng
(`extract_entity`, xem pipeline.py) để chặn thói quen model hay nhắc lại
tiêu đề/phần dẫn của snippet trước khi vào nội dung thật (xem README case
"cà phê": extract_unconditional + cắt n_predict ngắn không sửa được việc
này, chỉ cắt cụt giữa chừng phần lan man).

Cache file riêng (.link_cache_entity.json) - _cache_key không có `mode`
trong key, dùng chung .link_cache.json với adaptive.py sẽ đọc nhầm cache của
mode khác khi trùng question/snippet/n_predict/temperature.

Các phần không phụ thuộc mode (cluster theo answer, bias theo domain/nội
dung) tái dùng thẳng từ adaptive.py, không copy lại.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from adaptive import _best_cluster_result, _cache_key
from pipeline import N_PARALLEL, build_user_prompt, run_model
from score import score_answer
from search import search

CACHE_PATH = Path(__file__).parent / ".link_cache_entity.json"


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _process_link_entity(
    question: str, snippet: dict, n_predict: int, temperature: float, per_link_timeout_s: float,
) -> tuple[str, str, float] | None:
    user_prompt = build_user_prompt(question, [snippet])
    t0 = time.monotonic()
    try:
        result = run_model(
            user_prompt, mode="extract_entity", temperature=temperature,
            n_predict=n_predict, timeout=per_link_timeout_s,
        )
    except requests.exceptions.RequestException:
        return None
    elapsed = time.monotonic() - t0
    answer = result["answer"]
    verdict = score_answer(question, answer, [snippet])["verdict"]
    return answer, verdict, elapsed


def answer_entity_per_link(
    question: str,
    max_n: int = 8,
    consensus_threshold: float = 0.75,
    cluster_sim_threshold: float = 0.7,
    content_sim_threshold: float = 0.5,
    min_votes: int = 4,
    temperature: float = 0.0,
    n_predict: int = 20,
    per_link_timeout_s: float = 20,
    timeout_s: float = 90,
    use_cache: bool = True,
    debug: bool = False,
) -> dict:
    deadline = time.monotonic() + timeout_s
    snippets = search(question, n=max_n)
    if debug:
        print(f"[debug][entity] search: {len(snippets)} link cho {question!r}", flush=True)

    cache = _load_cache() if use_cache else {}
    answers: list[tuple[str, str]] = []
    used_snippets: list[dict] = []
    n_attempted = 0

    for batch_start in range(0, len(snippets), N_PARALLEL):
        if time.monotonic() >= deadline:
            if debug:
                print(f"[debug][entity] hết timeout_s={timeout_s}s trước link {batch_start + 1}, dừng", flush=True)
            break
        batch = list(enumerate(snippets[batch_start:batch_start + N_PARALLEL], start=batch_start + 1))
        n_attempted += len(batch)

        to_run = []
        for idx, snippet in batch:
            key = _cache_key(question, snippet, n_predict, temperature)
            cached = cache.get(key) if use_cache else None
            if cached:
                answer, verdict = cached["answer"], cached["verdict"]
                answers.append((answer, verdict))
                used_snippets.append(snippet)
                if debug:
                    print(f"[debug][entity] link {idx}/{len(snippets)} (cache) verdict={verdict}: {answer[:80]!r}", flush=True)
            else:
                to_run.append((idx, snippet, key))

        if to_run:
            with ThreadPoolExecutor(max_workers=len(to_run)) as executor:
                futures = {
                    executor.submit(_process_link_entity, question, snippet, n_predict, temperature, per_link_timeout_s):
                        (idx, snippet, key)
                    for idx, snippet, key in to_run
                }
                for future in as_completed(futures):
                    idx, snippet, key = futures[future]
                    result = future.result()
                    if result is None:
                        if debug:
                            print(f"[debug][entity] link {idx}/{len(snippets)} bỏ qua (timeout/lỗi)", flush=True)
                        continue
                    answer, verdict, elapsed = result
                    if use_cache:
                        cache[key] = {"answer": answer, "verdict": verdict}
                    answers.append((answer, verdict))
                    used_snippets.append(snippet)
                    if debug:
                        preview = answer[:80].replace("\n", " ")
                        print(
                            f"[debug][entity] link {idx}/{len(snippets)} ({elapsed:.1f}s) verdict={verdict} "
                            f"src={snippet.get('title', '')[:50]!r}\n               -> {preview!r}",
                            flush=True,
                        )
            if use_cache:
                _save_cache(cache)

        if len(answers) < min_votes:
            continue
        best = _best_cluster_result(question, answers, used_snippets, cluster_sim_threshold, content_sim_threshold)
        if debug and best:
            print(
                f"[debug][entity]   đồng thuận: {best['agreement']:.2f} (ngưỡng {consensus_threshold}) "
                f"coverage={best['cluster_size'] / n_attempted:.2f} bias={best['bias']} "
                f"content_bias={best['content_bias']}",
                flush=True,
            )
        if best and best["agreement"] >= consensus_threshold:
            if debug:
                print(f"[debug][entity] đạt ngưỡng sau {len(answers)} link, dừng sớm", flush=True)
            return {
                "question": question, "answer": best["answer"],
                "verdict": score_answer(question, best["answer"], used_snippets)["verdict"],
                "n_links_used": len(answers), "agreement": round(best["agreement"], 2),
                "coverage": round(best["cluster_size"] / n_attempted, 2), "bias": best["bias"],
                "content_bias": best["content_bias"],
                "confident": True, "snippets": used_snippets,
            }

    best = _best_cluster_result(question, answers, used_snippets, cluster_sim_threshold, content_sim_threshold)
    if debug:
        print(
            f"[debug][entity] hết link/timeout sau {len(answers)} link, "
            f"agreement={best['agreement']:.2f}" if best else "[debug][entity] không có mẫu nào dùng được",
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
    from pathlib import Path as _Path

    import pipeline

    pipeline.MODEL_PATH = _Path(__file__).parent / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

    q = " ".join(sys.argv[1:]) or "nước sản xuất nhiều cà phê nhất thế giới"
    t0 = time.monotonic()
    out = answer_entity_per_link(q, debug=True)
    elapsed = time.monotonic() - t0

    print(f"Câu hỏi: {q}")
    print(f"Trả lời: {out['answer']}")
    print(
        f"verdict={out['verdict']} confident={out.get('confident')} "
        f"n_links_used={out.get('n_links_used')} agreement={out.get('agreement')} "
        f"coverage={out.get('coverage')} bias={out.get('bias')} content_bias={out.get('content_bias')} "
        f"elapsed={elapsed:.1f}s"
    )
