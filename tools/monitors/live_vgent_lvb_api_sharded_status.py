from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path("results/vgent/rag_api_qwen2vl7b_cached_graph_v1")
RUN_PREFIX = "full1337_graph_fair16_shard"
OUT = ROOT / "full1337_graph_fair16_sharded_live_status.md"
TMUX_PREFIX = "vgent_lvb_qwen2_api_shard"
TOTAL = 1337


def _row_stats(path: Path) -> tuple[int, int]:
    rows = path / "rows.jsonl"
    if not rows.exists():
        return 0, 0
    done = 0
    correct = 0
    with rows.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            done += 1
            try:
                correct += int(bool(json.loads(line).get("correct")))
            except Exception:
                pass
    return done, correct


def _start_ts(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    pattern = re.compile(r"^\[(.*?)\] start total=")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            except ValueError:
                return None
    return None


def _tmux_active(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _fmt(seconds: float | None) -> str:
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


def _tail(path: Path, n: int = 8) -> str:
    log = path / "run.log"
    if not log.exists():
        return ""
    return "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


def main() -> None:
    shard_dirs = sorted(path for path in ROOT.glob(f"{RUN_PREFIX}*") if path.is_dir())
    now = time.time()
    rows = []
    total_done = 0
    total_correct = 0
    earliest_start = None

    for path in shard_dirs:
        suffix = path.name.replace(RUN_PREFIX, "")
        shard_id = suffix.lstrip("_") or "?"
        done, correct = _row_stats(path)
        total_done += done
        total_correct += correct
        start = _start_ts(path / "run.log")
        if start is not None:
            earliest_start = start if earliest_start is None else min(earliest_start, start)
        active = _tmux_active(f"{TMUX_PREFIX}{shard_id}")
        rows.append((shard_id, active, done, correct, path))

    elapsed = now - earliest_start if earliest_start else None
    qps = total_done / elapsed if elapsed and elapsed > 0 else None
    eta = (TOTAL - total_done) / qps if qps and total_done < TOTAL else 0 if total_done >= TOTAL else None
    acc = total_correct / total_done if total_done else None

    lines = [
        "# Live Vgent LVB API Sharded Status",
        "",
        f"Updated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| completed | `{total_done}/{TOTAL}` |",
        f"| correct | `{total_correct}` |",
        f"| accuracy | `{acc:.4f}` |" if acc is not None else "| accuracy | `n/a` |",
        f"| elapsed | `{_fmt(elapsed)}` |",
        f"| throughput | `{qps:.3f} q/s` |" if qps is not None else "| throughput | `n/a` |",
        f"| ETA | `{_fmt(eta)}` |",
        "",
        "## Shards",
        "",
        "| Shard | Active | Done | Correct | Accuracy |",
        "|---:|---:|---:|---:|---:|",
    ]
    for shard_id, active, done, correct, _ in rows:
        shard_acc = correct / done if done else None
        lines.append(
            f"| {shard_id} | `{str(active).lower()}` | {done} | {correct} | {shard_acc:.4f} |"
            if shard_acc is not None
            else f"| {shard_id} | `{str(active).lower()}` | {done} | {correct} | n/a |"
        )

    lines.append("")
    lines.append("## Log Tails")
    for shard_id, _, _, _, path in rows:
        lines.extend(["", f"### Shard {shard_id}", "", "```text", _tail(path), "```"])
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
