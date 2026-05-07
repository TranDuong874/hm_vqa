# Tools

Runnable utilities that are not core experiment runners.

## Layout

- `tools/monitors/`
  - Live matrix/status writers for HD-EPIC, LongVideoBench, and Vgent runs.
- `tools/vgent/`
  - Vgent API probes and cache post-processing utilities.

Run tools from the repository root with:

```bash
PYTHONPATH=.:src .venv/bin/python tools/<group>/<script>.py
```

Long-running monitors can be repeated without shell wrapper files:

```bash
while true; do
  PYTHONPATH=.:src .venv/bin/python tools/monitors/live_hd_epic_fgal24_all_matrix.py
  sleep 2
done
```
