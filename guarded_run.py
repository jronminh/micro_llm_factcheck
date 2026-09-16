"""Chay 1 script test cu (explore_*.py, chain.py, benchmark.py...) duoi giam
sat nhiet - doi may nguoi truoc khi bat dau, kill llama-server + dung
subprocess ngay neu nhiet vuot nguong lien tuc qua lau, khong doi script con
chay xong. Khong sua tung script cu, wrapper ben ngoai.

Dung: python3 guarded_run.py chain.py
      python3 guarded_run.py explore_adaptive.py
"""
import subprocess
import sys
import threading

from thermal import ThermalGuard


def main() -> int:
    if len(sys.argv) < 2:
        print("dung: python3 guarded_run.py <script.py> [args...]")
        return 2

    guard = ThermalGuard()
    guard.wait_for_cooldown()
    guard.start()

    proc = subprocess.Popen([sys.executable, *sys.argv[1:]])

    def _watch() -> None:
        guard.abort.wait()
        print(f"[guarded_run] kill switch: {guard.reason} - dung subprocess pid={proc.pid}", flush=True)
        proc.terminate()

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()

    try:
        return proc.wait()
    finally:
        guard.stop()


if __name__ == "__main__":
    sys.exit(main())
