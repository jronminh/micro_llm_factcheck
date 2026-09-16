"""So sánh Qwen3-0.6B (voi /no_think, xem notes/micro-llm-alternatives-to-qwen.md)
với Qwen2.5-0.5B trên đúng bộ 8 câu hỏi khó của explore_adaptive.py.

Monkey-patch adaptive.build_user_prompt (không phải llm.build_user_prompt
- adaptive.py đã `from llm import build_user_prompt` nên giữ tham chiếu
riêng) để nối "/no_think" vào cuối prompt gửi model, KHÔNG đụng vào câu hỏi
dùng để search (question truyền cho search() và build_user_prompt() là cùng
1 biến trong adaptive.py - phải patch ở tầng prompt, không phải tầng câu
hỏi, nếu không sẽ làm bẩn query search).
"""
import time
from pathlib import Path

import adaptive
import llm
from adaptive import answer_question_per_link
from explore_adaptive import QUESTIONS

llm.MODEL_PATH = Path(__file__).parent / "models" / "qwen3-0.6b-q4_k_m.gguf"

_original_build_user_prompt = adaptive.build_user_prompt


def _build_user_prompt_no_think(question: str, snippets: list[dict]) -> str:
    return _original_build_user_prompt(question, snippets) + " /no_think"


adaptive.build_user_prompt = _build_user_prompt_no_think

if __name__ == "__main__":
    rows = []
    for kind, q in QUESTIONS:
        t0 = time.monotonic()
        out = answer_question_per_link(q, debug=False)
        elapsed = time.monotonic() - t0
        rows.append((kind, q, out, elapsed))
        print(f"[{kind}] {q}")
        print(f"  -> {out['answer']!r}")
        print(
            f"  verdict={out['verdict']} confident={out.get('confident')} "
            f"n_links_used={out.get('n_links_used')} agreement={out.get('agreement')} "
            f"coverage={out.get('coverage')} bias={out.get('bias')} content_bias={out.get('content_bias')} "
            f"elapsed={elapsed:.1f}s"
        )
        print()

    print(f"{'=' * 10} tổng kết {'=' * 10}")
    tally = {}
    for _, _, out, _ in rows:
        tally[out["verdict"]] = tally.get(out["verdict"], 0) + 1
    print(", ".join(f"{v}={n}" for v, n in sorted(tally.items())))
    avg_elapsed = sum(r[3] for r in rows) / len(rows)
    print(f"elapsed avg={avg_elapsed:.1f}s")
