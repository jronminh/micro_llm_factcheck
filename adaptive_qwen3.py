"""Fork của adaptive.py, dùng riêng cho model reasoning kiểu Qwen3 (dual-mode
<think>). Thay vì 1 lần gọi extract_unconditional như adaptive.py gốc, mỗi
link chạy qua 3 bước, mỗi bước 1 system prompt riêng (llm.SYSTEM_PROMPTS
"qwen3_extract"/"qwen3_critique"/"qwen3_finalize"):

1. Trích xuất (bật reasoning) - đọc snippet, suy luận, đưa ra câu trả lời
   ứng viên kèm trích dẫn.
2. Phản biện (bật reasoning) - vai trò reviewer độc lập, chỉ thấy đoạn trích
   gốc + câu trả lời ứng viên (KHÔNG thấy reasoning_content của bước 1), suy
   luận xem câu trả lời có căn cứ thật hay suy diễn/bịa, kết luận HOP_LE/
   KHONG_HOP_LE.
3. Chốt (bật reasoning nhẹ) - cho câu trả lời ứng viên + kết quả phản biện,
   in ra câu trả lời cuối hoặc "không có thông tin" nếu KHONG_HOP_LE.

ĐỒNG BỘ THEO BƯỚC (batch-lockstep), không phải mỗi link tự chạy hết 3 bước
độc lập: tất cả link trong batch chạy xong bước 1 mới cùng sang bước 2, xong
bước 2 mới cùng sang bước 3. Lý do đổi từ thiết kế ban đầu (mỗi link tự chạy
hết chain, xem git history): llama-server chọn slot dựa trên độ giống prompt
với lần trước đó trong slot ("LCP similarity", thấy trong .server.log) - để
1 slot xử lý liên tiếp cùng 1 loại bước (cùng system prompt) tận dụng cache
đó tốt hơn so với 1 slot phải luân phiên giữa các loại bước khác nhau của
các link không đồng bộ với nhau. CHƯA đo được lockstep có thực sự nhanh hơn
non-lockstep hay không (cả 2 đều đã xác nhận: song song nhanh hơn tuần tự,
xem notes/micro-llm-alternatives-to-qwen.md) - đáng so sánh thêm nếu cần.

"Độc lập" ở bước phản biện là 1 lần gọi HTTP tách biệt (không phải 1
conversation nhiều turn) - chỉ thấy `content` (câu trả lời đã chốt) của bước
trước, không thấy `reasoning_content` (quá trình suy nghĩ thô) - giống 1
reviewer con người chỉ đọc bản đã chốt, không cần đọc hết bản nháp.

KHÔNG dùng /no_think ở đây (khác cách dùng Qwen3 ở nơi khác trong dự án) -
mục đích ngược lại: tận dụng reasoning để tự lọc, không tắt nó đi. Đổi lại
mỗi link giờ tốn 3 lần gọi model thay vì 1, cộng thêm overhead reasoning của
Qwen3 (đã đo: chậm hơn Qwen2.5-0.5B ~2.2 lần cho 1 lần gọi, xem
notes/micro-llm-alternatives-to-qwen.md).

CẦN LƯU Ý (chưa giải quyết): trong mọi lần test tới giờ, bước phản biện luôn
kết luận HOP_LE - chưa từng bác bỏ câu nào, vì mọi câu hỏi test đều là câu dễ
mà bước 1 đã trả lời đúng. Chưa có bằng chứng bước phản biện thực sự lọc
được gì - cần test với câu hỏi khó (nơi bước 1 nhiều khả năng sai/bịa) mới
biết prompt "qwen3_critique" có tác dụng thật hay chỉ rubber-stamp.

Cache file riêng (.link_cache_qwen3.json) - không dùng chung với adaptive.py
(1 lần gọi) hay adaptive_entity.py (1 lần gọi, mode khác) vì đây là kết quả
của cả chain 3 bước, không so sánh được với answer của 1 lần gọi đơn. Cache
key cũng không tái dùng adaptive._cache_key thẳng (chỉ nhận 1 n_predict) - tự
viết _cache_key_qwen3 gộp cả 3 n_predict, tránh đúng loại bug đã gặp (thiếu
MODEL_PATH trong key làm 4/8 câu bị đọc nhầm cache của model khác, xem
adaptive._cache_key).
"""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

import llm
from adaptive import _best_cluster_result
from llm import N_PARALLEL, build_user_prompt, run_model
from score import score_answer
from search import search

CACHE_PATH = Path(__file__).parent / ".link_cache_qwen3.json"


def _cache_key_qwen3(
    question: str, snippet: dict, n_predict_extract: int, n_predict_critique: int,
    n_predict_finalize: int, temperature: float,
) -> str:
    raw = (
        f"{question}||{n_predict_extract}||{n_predict_critique}||{n_predict_finalize}||"
        f"{temperature}||{llm.MODEL_PATH}||{snippet.get('url') or snippet.get('snippet', '')}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _build_critique_prompt(question: str, snippet: dict, candidate_answer: str) -> str:
    return (
        f"[Đoạn trích gốc]\n{snippet.get('title', '')}: {snippet.get('snippet', '')}\n\n"
        f"[Câu hỏi]\n{question}\n\n"
        f"[Câu trả lời ứng viên]\n{candidate_answer}"
    )


def _build_finalize_prompt(question: str, candidate_answer: str, critique: str) -> str:
    return (
        f"[Câu hỏi]\n{question}\n\n"
        f"[Câu trả lời ứng viên]\n{candidate_answer}\n\n"
        f"[Kết quả kiểm tra]\n{critique}"
    )


def _step_debug(step_name: str, result: dict) -> dict:
    # finish_reason="length" o day la dau hieu chinh de chan doan: model bi
    # cat cut boi n_predict truoc khi tu ket luan (co the van dang o giua
    # <think>) - phan biet voi "stop" (tu dung dung luc). reasoning_len ~4
    # ky tu/token, chi la uoc luong tho de xem buoc nao "nghi" nhieu nhat.
    reasoning = result.get("reasoning") or ""
    return {
        "step": step_name,
        "elapsed_s": round(result.get("elapsed_s", 0.0), 1),
        "finish_reason": result.get("finish_reason"),
        "completion_tokens": result.get("completion_tokens"),
        "gen_tps": result.get("gen_tps"),
        "reasoning_len_chars": len(reasoning),
        "answer_preview": result["answer"][:80].replace("\n", " "),
    }


def _run_step_batch(
    mode: str, items: list[tuple[int, str]], n_predict: int, temperature: float,
    per_link_timeout_s: float, parallel: bool,
) -> dict[int, dict | None]:
    """Chạy 1 bước cho cả batch link ĐỒNG BỘ - tất cả item ở đây cùng thuộc
    1 bước, gửi song song (hoặc tuần tự nếu parallel=False, để debug/so
    sánh). Trả dict idx -> kết quả (None nếu timeout/lỗi HTTP) - không raise,
    để 1 link lỗi không kéo cả batch dừng giữa chừng."""

    def _call(prompt: str) -> dict:
        t0 = time.monotonic()
        result = run_model(prompt, mode=mode, temperature=temperature, n_predict=n_predict, timeout=per_link_timeout_s)
        result["elapsed_s"] = time.monotonic() - t0
        return result

    results: dict[int, dict | None] = {}
    max_workers = len(items) if parallel else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_call, prompt): idx for idx, prompt in items}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except requests.exceptions.RequestException:
                results[idx] = None
    return results


def answer_question_qwen3_per_link(
    question: str,
    max_n: int = 8,
    consensus_threshold: float = 0.75,
    cluster_sim_threshold: float = 0.7,
    content_sim_threshold: float = 0.5,
    min_votes: int = 4,
    temperature: float = 0.0,
    n_predict_extract: int = 400,
    n_predict_critique: int = 300,
    n_predict_finalize: int = 200,
    per_link_timeout_s: float = 60,
    timeout_s: float = 300,
    use_cache: bool = True,
    debug: bool = False,
    parallel: bool = True,
) -> dict:
    deadline = time.monotonic() + timeout_s
    snippets = search(question, n=max_n)
    if debug:
        print(f"[debug][qwen3] search: {len(snippets)} link cho {question!r}", flush=True)

    cache = _load_cache() if use_cache else {}
    answers: list[tuple[str, str]] = []
    used_snippets: list[dict] = []
    n_attempted = 0

    for batch_start in range(0, len(snippets), N_PARALLEL):
        if time.monotonic() >= deadline:
            if debug:
                print(f"[debug][qwen3] hết timeout_s={timeout_s}s trước link {batch_start + 1}, dừng", flush=True)
            break
        batch = list(enumerate(snippets[batch_start:batch_start + N_PARALLEL], start=batch_start + 1))
        n_attempted += len(batch)

        to_run = []
        for idx, snippet in batch:
            key = _cache_key_qwen3(
                question, snippet, n_predict_extract, n_predict_critique, n_predict_finalize, temperature,
            )
            cached = cache.get(key) if use_cache else None
            if cached:
                answer, verdict = cached["answer"], cached["verdict"]
                answers.append((answer, verdict))
                used_snippets.append(snippet)
                if debug:
                    print(f"[debug][qwen3] link {idx}/{len(snippets)} (cache) verdict={verdict}: {answer[:80]!r}", flush=True)
            else:
                to_run.append((idx, snippet, key))

        if to_run:
            # Dong bo theo buoc (lockstep) - xem docstring dau file ve ly do
            # (LCP slot cache cua llama-server). snippet_by_idx/key_by_idx giu
            # nguyen thu tu link qua ca 3 pha; alive_* rut dan neu co link
            # timeout/loi o pha truoc - khong ep pha sau phai cho link da chet.
            snippet_by_idx = {idx: snippet for idx, snippet, _ in to_run}
            key_by_idx = {idx: key for idx, snippet, key in to_run}

            extract_items = [(idx, build_user_prompt(question, [snippet_by_idx[idx]])) for idx in snippet_by_idx]
            extract_results = _run_step_batch(
                "qwen3_extract", extract_items, n_predict_extract, temperature, per_link_timeout_s, parallel,
            )
            alive = [idx for idx in snippet_by_idx if extract_results.get(idx) is not None]
            if debug:
                n_dropped = len(snippet_by_idx) - len(alive)
                print(
                    f"[debug][qwen3] batch {len(snippet_by_idx)} link - xong bước extract"
                    + (f", {n_dropped} link bỏ qua (timeout/lỗi)" if n_dropped else ""),
                    flush=True,
                )

            critique_items = [
                (idx, _build_critique_prompt(question, snippet_by_idx[idx], extract_results[idx]["answer"]))
                for idx in alive
            ]
            critique_results = _run_step_batch(
                "qwen3_critique", critique_items, n_predict_critique, temperature, per_link_timeout_s, parallel,
            )
            alive2 = [idx for idx in alive if critique_results.get(idx) is not None]
            if debug:
                n_dropped = len(alive) - len(alive2)
                print(
                    f"[debug][qwen3] batch {len(alive)} link - xong bước critique"
                    + (f", {n_dropped} link bỏ qua (timeout/lỗi)" if n_dropped else ""),
                    flush=True,
                )

            finalize_items = [
                (idx, _build_finalize_prompt(question, extract_results[idx]["answer"], critique_results[idx]["answer"]))
                for idx in alive2
            ]
            finalize_results = _run_step_batch(
                "qwen3_finalize", finalize_items, n_predict_finalize, temperature, per_link_timeout_s, parallel,
            )

            for idx in alive2:
                finalize_r = finalize_results.get(idx)
                if finalize_r is None:
                    if debug:
                        print(f"[debug][qwen3] link {idx}/{len(snippets)} bỏ qua sau bước finalize (timeout/lỗi)", flush=True)
                    continue
                snippet, key = snippet_by_idx[idx], key_by_idx[idx]
                answer = finalize_r["answer"]
                verdict = score_answer(question, answer, [snippet])["verdict"]
                if use_cache:
                    cache[key] = {"answer": answer, "verdict": verdict}
                answers.append((answer, verdict))
                used_snippets.append(snippet)
                if debug:
                    preview = answer[:80].replace("\n", " ")
                    print(
                        f"[debug][qwen3] link {idx}/{len(snippets)} verdict={verdict} "
                        f"src={snippet.get('title', '')[:50]!r}\n              -> {preview!r}",
                        flush=True,
                    )
                    for step_name, r in [
                        ("extract", extract_results[idx]), ("critique", critique_results[idx]), ("finalize", finalize_r),
                    ]:
                        sd = _step_debug(step_name, r)
                        print(
                            f"[debug][qwen3]     {sd['step']:<9} {sd['elapsed_s']:>5.1f}s "
                            f"finish={sd['finish_reason']} completion_tokens={sd['completion_tokens']} "
                            f"gen_tps={sd['gen_tps']} reasoning_len_chars={sd['reasoning_len_chars']} "
                            f"-> {sd['answer_preview']!r}",
                            flush=True,
                        )
            if use_cache:
                _save_cache(cache)

        if len(answers) < min_votes:
            continue
        best = _best_cluster_result(question, answers, used_snippets, cluster_sim_threshold, content_sim_threshold)
        if debug and best:
            print(
                f"[debug][qwen3]   đồng thuận: {best['agreement']:.2f} (ngưỡng {consensus_threshold}) "
                f"coverage={best['cluster_size'] / n_attempted:.2f} bias={best['bias']} "
                f"content_bias={best['content_bias']}",
                flush=True,
            )
        if best and best["agreement"] >= consensus_threshold:
            if debug:
                print(f"[debug][qwen3] đạt ngưỡng sau {len(answers)} link, dừng sớm", flush=True)
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
            f"[debug][qwen3] hết link/timeout sau {len(answers)} link, "
            f"agreement={best['agreement']:.2f}" if best else "[debug][qwen3] không có mẫu nào dùng được",
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

    llm.MODEL_PATH = Path(__file__).parent / "models" / "qwen3-0.6b-q4_k_m.gguf"

    # --sequential: gui request tung link mot thay vi song song - de so sanh
    # wall-clock that voi che do song song mac dinh (xem comment o
    # answer_question_qwen3_per_link ve tai sao gia dinh "song song luon
    # nhanh hon" dang bi nghi ngo cho chain 3 buoc nay).
    parallel = "--sequential" not in sys.argv
    args = [a for a in sys.argv[1:] if a != "--sequential"]

    # max_n=3 mac dinh khi chay truc tiep - quy mo nho de do chi phi that
    # (3 buoc/link, chua biet tong thoi gian/nhiet) truoc khi thu max_n=8 nhu
    # adaptive.py goc.
    q = " ".join(args) or "Thủ đô Việt Nam là gì?"
    t0 = time.monotonic()
    out = answer_question_qwen3_per_link(q, max_n=3, debug=True, parallel=parallel)
    elapsed = time.monotonic() - t0

    print(f"Câu hỏi: {q}")
    print(f"Trả lời: {out['answer']}")
    print(
        f"verdict={out['verdict']} confident={out.get('confident')} "
        f"n_links_used={out.get('n_links_used')} agreement={out.get('agreement')} "
        f"coverage={out.get('coverage')} bias={out.get('bias')} content_bias={out.get('content_bias')} "
        f"elapsed={elapsed:.1f}s"
    )
