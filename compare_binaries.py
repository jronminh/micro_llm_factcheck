"""So sanh end-to-end ban `pkg install` vs ban build tu source cua llama-server
tren llm that (khong phai llama-bench tho, da co so lieu trong README).

Truoc gio LLAMA_SERVER_BIN trong llm.py tro san sang build-tu-source va
chua tung bi doi qua lai trong 1 lan chay - bug _stop_server() (pkill -x
khong khop duoc process, xem llm.py) co nghia neu doi qua lai truoc day
co the da am tham benchmark nham binary cu. Sau khi sua _stop_server()/
ensure_server() de tu parse ps aux + check ca LLAMA_SERVER_BIN, script nay la
lan dau so sanh that giua 2 binary tren cau hoi thuc.

Chay 2 pha: don gian truoc (rui ro thap, phat hien loi/crash som), phuc tap
sau (giong bo cau hoi kho da dung tune adaptive.py - nhieu link, prompt dai,
tai CPU nang/lau hon that su). Chi sang pha phuc tap neu pha don gian chay
xong sach, khong bi kill switch cat.

Kill switch nhiet: xem thermal.py (PMIC la tin hieu chinh, CPU-die la failsafe
phu) - dung server ngay lap tuc (khong doi cau hoi dang chay xong) va bao abort
khi kich hoat.
"""
import csv
import time
from pathlib import Path

import requests

import llm
from thermal import ThermalGuard

RESULTS_CSV = Path(__file__).parent / "benchmark" / "binary_compare.csv"

BINARIES = {
    "pkg": Path("/data/data/com.termux/files/usr/bin/llama-server"),
    "source": Path.home() / "vendor" / "llama.cpp" / "build" / "bin" / "llama-server",
}

# Giu nguyen bo cau hoi don gian tu compare_models.py de nhat quan.
SIMPLE_QUESTIONS = [
    "Chieu cao nui Everest la bao nhieu?",
    "Ai la tong thong My hien tai?",
    "Python la ngon ngu lap trinh gi?",
]

# 3 cau multi-hop/so sanh kho da dung tune adaptive.py (xem README, muc "day
# den gioi han") - can nhieu link/prompt dai de mo phong tai CPU that su, uu
# tien dau tot cho search tieng Viet.
COMPLEX_QUESTIONS = [
    "Thủ đô nước láng giềng phía Bắc Việt Nam là gì?",
    "Tháp Eiffel và tháp Tokyo Tower, cái nào cao hơn?",
    "GDP bình quân đầu người Việt Nam năm 2025 so với năm 2015 tăng bao nhiêu?",
]

def _run_questions(label: str, questions: list[str], guard: ThermalGuard) -> list[dict]:
    rows = []
    for q in questions:
        if guard.abort.is_set():
            break
        t0 = time.monotonic()
        try:
            out = llm.answer_question(q)
        except requests.exceptions.RequestException:
            if guard.abort.is_set():
                print(f"  [{label}] request loi do server bi kill switch dung - bo qua cau con lai")
                break
            raise
        elapsed = time.monotonic() - t0
        preview = out["answer"][:120].replace("\n", " ")
        rows.append({
            "binary": label,
            "question": q,
            "total_s": round(elapsed, 2),
            "gen_tps": out["gen_tps"],
            "answer_preview": preview,
        })
        print(f"  [{label}] {elapsed:.1f}s gen_tps={out['gen_tps']} :: {q} -> {preview}")
    return rows


def run_phase(phase_name: str, questions: list[str], guard: ThermalGuard) -> list[dict]:
    print(f"\n== Pha {phase_name} ==")
    all_rows = []
    for label, bin_path in BINARIES.items():
        if guard.abort.is_set():
            break
        if not bin_path.exists():
            print(f"Bo qua binary {label}: khong thay {bin_path}")
            continue
        print(f"-- binary={label} ({bin_path}) --")
        llm.LLAMA_SERVER_BIN = bin_path
        llm.ensure_server()
        all_rows.extend(_run_questions(label, questions, guard))
    return all_rows


if __name__ == "__main__":
    guard = ThermalGuard()
    guard.wait_for_cooldown()
    guard.start()

    all_rows = []
    try:
        all_rows.extend(run_phase("don_gian", SIMPLE_QUESTIONS, guard))
        if guard.abort.is_set():
            print(f"\nDUNG SOM sau pha don_gian: {guard.reason}")
        else:
            all_rows.extend(run_phase("phuc_tap", COMPLEX_QUESTIONS, guard))
            if guard.abort.is_set():
                print(f"\nDUNG SOM giua pha phuc_tap: {guard.reason}")
    finally:
        guard.stop()

    RESULTS_CSV.parent.mkdir(exist_ok=True)
    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["binary", "question", "total_s", "gen_tps", "answer_preview"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'binary':<8} {'total_s':>8} {'gen_tps':>8}  question")
    for r in all_rows:
        print(f"{r['binary']:<8} {r['total_s']:>8} {r['gen_tps']:>8}  {r['question']}")
    print(f"\nDa ghi ket qua vao {RESULTS_CSV}")
