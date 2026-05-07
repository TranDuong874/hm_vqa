from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


RUN_DIR = Path("results/vgent/rag_api_qwen2vl7b_cached_graph_v1/full1337_graph_fair16")
ROWS = RUN_DIR / "rows.jsonl"
LOG = RUN_DIR / "run.log"
OUT = RUN_DIR / "live_status.md"
TMUX_SESSION = "vgent_lvb_qwen2_api_full1337"
TOTAL = 1337


def _count_rows() -> tuple[int, int]:
    if not ROWS.exists():
        return 0, 0
    done = 0
    correct = 0
    with ROWS.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            done += 1
            try:
                correct += int(bool(json.loads(line).get("correct")))
            except Exception:
                pass
    return done, correct


def _tmux_active() -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", TMUX_SESSION],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _first_start_ts() -> float | None:
    if not LOG.exists():
        return None
    pattern = re.compile(r"^\[(.*?)\] start total=")
    for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            except ValueError:
                return None
    return None


def _tail_log(n: int = 30) -> str:
    if not LOG.exists():
        return ""
    lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def _fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def main() -> None:
    done, correct = _count_rows()
    active = _tmux_active()
    now = time.time()
    start = _first_start_ts()
    elapsed = (now - start) if start else None
    qps = (done / elapsed) if elapsed and elapsed > 0 else None
    remaining = max(TOTAL - done, 0)
    eta = (remaining / qps) if qps and qps > 0 else None
    accuracy = (correct / done) if done else None

    lines = [
        "# Live Vgent LVB API Status",
        "",
        f"Updated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        f"Run dir: `{RUN_DIR}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| tmux active | `{str(active).lower()}` |",
        f"| completed | `{done}/{TOTAL}` |",
        f"| correct | `{correct}` |",
        f"| accuracy | `{accuracy:.4f}` |" if accuracy is not None else "| accuracy | `n/a` |",
        f"| elapsed | `{_fmt_eta(elapsed)}` |",
        f"| throughput | `{qps:.3f} q/s` |" if qps is not None else "| throughput | `n/a` |",
        f"| ETA | `{_fmt_eta(eta)}` |",
        "",
        "## Log Tail",
        "",
        "```text",
        _tail_log(),
        "```",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
