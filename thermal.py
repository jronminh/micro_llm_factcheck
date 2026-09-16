"""Kill switch nhiet 2 tang, dung chung cho moi script test nang
(compare_binaries.py, guarded_run.py).

Lan dau chi doc zone `cpuss-*`/`cpu-0-*`/`cpu-1-*` (cam bien junction ngay
tai nhan CPU) va set nguong 60C - hoa ra sai loai cam bien: junction CPU chay
len 80-95C khi tai nang la BINH THUONG tren hau het SoC di dong (khong rieng
gi Snapdragon 8 Gen1), ban than SoC da tu ha xung (DVFS) truoc khi cham TJmax
(~100-115C tuy chip) - dung no lam kill switch chinh nghia la bao dong gia
gan nhu ngay khi model chay, bat ke may co thuc su "nong" theo nghia an toan
hay khong (da thay: 3 lan test lien tiep deu cham 60C trong <15s).

Doi chieu voi zone PMIC (`pm8350_tz`/`pm8350c_tz`) va pin (qua
termux-battery-status) do song song cung luc: PMIC/pin tang/giam cham hon
nhieu va o muc thap hon han (37-45C ngay ca khi CPU-die dang o 90C) - day moi
la chi so bam sat "may nong khi cam tay"/an toan pin, dung ngon ngu thuong
ngay muon noi khi noi "nhiet do qua cao".

Thiet ke lai: PMIC la tin hieu CHINH (lien quan an toan/pin that), CPU-die
la failsafe PHU voi nguong sat TJmax (chi kich khi co gi that bat thuong, vi
SoC da tu bao ve no truoc do roi).

Chi tiet nghien cuu/nguon tham khao: notes/android-thermal-apis.md.
"""
import re
import subprocess
import threading
import time
from pathlib import Path

PMIC_ZONE_RE = re.compile(r"^pm8350c?_tz$")
CPU_ZONE_RE = re.compile(r"^(cpuss-|cpu-0-|cpu-1-)\d+$")

PMIC_THRESHOLD_C = 50.0
PMIC_SUSTAINED_SECONDS = 180.0
CPU_FAILSAFE_THRESHOLD_C = 100.0
CPU_FAILSAFE_SUSTAINED_SECONDS = 60.0
POLL_INTERVAL_S = 5.0
COOLDOWN_BEFORE_START_C = 45.0  # theo PMIC, khong phai CPU-die


def _zones_matching(pattern: re.Pattern) -> list[Path]:
    zones = []
    for type_path in Path("/sys/class/thermal").glob("thermal_zone*/type"):
        try:
            name = type_path.read_text().strip()
        except OSError:
            continue
        if pattern.match(name):
            zones.append(type_path.parent / "temp")
    return zones


def max_temp_c(zones: list[Path]) -> float | None:
    readings = []
    for p in zones:
        try:
            readings.append(int(p.read_text().strip()) / 1000)
        except (OSError, ValueError):
            continue
    return max(readings) if readings else None


def kill_llama_server() -> None:
    # ps aux tu parse, khong dung pkill/pgrep - tren may nay 2 lenh do khong
    # khop duoc process du ps aux thay ro no (xem llm._llama_server_pids).
    out = subprocess.run(["ps", "aux"], capture_output=True, text=True, check=False).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) > 10 and parts[10].endswith("/llama-server"):
            subprocess.run(["kill", "-9", parts[1]], check=False)


class _SustainedTrigger:
    """Theo doi 1 chi so vuot nguong lien tuc bao lau - dung chung logic cho
    ca PMIC (chinh) va CPU-die (failsafe), khac nhau nguong/thoi gian."""

    def __init__(self, label: str, zones: list[Path], threshold_c: float, sustained_s: float):
        self.label = label
        self.zones = zones
        self.threshold_c = threshold_c
        self.sustained_s = sustained_s
        self._hot_since: float | None = None

    def check(self) -> str | None:
        temp = max_temp_c(self.zones)
        if temp is None:
            return None
        now = time.monotonic()
        if temp >= self.threshold_c:
            self._hot_since = self._hot_since or now
            elapsed = now - self._hot_since
            print(
                f"[thermal] {self.label} {temp:.1f}C >= {self.threshold_c}C, "
                f"lien tuc {elapsed:.0f}s/{self.sustained_s:.0f}s",
                flush=True,
            )
            if elapsed >= self.sustained_s:
                return f"{self.label} {temp:.1f}C vuot {self.threshold_c}C lien tuc >= {self.sustained_s:.0f}s"
        else:
            if self._hot_since is not None:
                print(f"[thermal] {self.label} {temp:.1f}C < nguong, reset dem lien tuc", flush=True)
            self._hot_since = None
        return None


class ThermalGuard:
    """Kill switch 2 tang: PMIC (pm8350_tz/pm8350c_tz) la tin hieu chinh, bam
    sat pin/board - lien quan thuc te den an toan khi cam may. CPU-die
    (cpuss-*/cpu-0-*/cpu-1-*) la failsafe phu, nguong sat TJmax, chi kich khi
    co gi that bat thuong (SoC da tu throttle truoc do roi)."""

    def __init__(
        self,
        pmic_threshold_c: float = PMIC_THRESHOLD_C,
        pmic_sustained_s: float = PMIC_SUSTAINED_SECONDS,
        cpu_threshold_c: float = CPU_FAILSAFE_THRESHOLD_C,
        cpu_sustained_s: float = CPU_FAILSAFE_SUSTAINED_SECONDS,
        interval_s: float = POLL_INTERVAL_S,
        cooldown_c: float = COOLDOWN_BEFORE_START_C,
    ):
        self.interval_s = interval_s
        self.cooldown_c = cooldown_c
        self.abort = threading.Event()
        self.reason = ""
        self._pmic_zones = _zones_matching(PMIC_ZONE_RE)
        self._pmic = _SustainedTrigger("PMIC", self._pmic_zones, pmic_threshold_c, pmic_sustained_s)
        self._cpu = _SustainedTrigger(
            "CPU-die-failsafe", _zones_matching(CPU_ZONE_RE), cpu_threshold_c, cpu_sustained_s
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def wait_for_cooldown(self) -> None:
        if not self._pmic_zones:
            return
        temp = max_temp_c(self._pmic_zones)
        if temp is not None and temp > self.cooldown_c:
            print(f"[thermal] PMIC dang {temp:.1f}C > {self.cooldown_c}C, doi nguoi truoc khi bat dau...", flush=True)
            while temp is not None and temp > self.cooldown_c:
                time.sleep(self.interval_s)
                temp = max_temp_c(self._pmic_zones)
            print(f"[thermal] PMIC da nguoi con {temp:.1f}C, tiep tuc", flush=True)

    def start(self) -> None:
        if not self._pmic_zones:
            print(
                "[thermal] CANH BAO: khong doc duoc zone PMIC nao - kill switch CHINH vo hieu, "
                "chi con failsafe CPU-die.",
                flush=True,
            )
        print(
            f"[thermal] guard bat dau - PMIC nguong {self._pmic.threshold_c}C/{self._pmic.sustained_s:.0f}s "
            f"(chinh), CPU-die failsafe {self._cpu.threshold_c}C/{self._cpu.sustained_s:.0f}s (phu)",
            flush=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            for trigger in (self._pmic, self._cpu):
                if not trigger.zones:
                    continue
                reason = trigger.check()
                if reason:
                    self.reason = reason
                    print(f"[thermal] KILL SWITCH: {reason} - dung llama-server ngay", flush=True)
                    kill_llama_server()
                    self.abort.set()
                    return
            self._stop.wait(self.interval_s)
