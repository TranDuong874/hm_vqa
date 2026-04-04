# HM-VQA

This repository was reset to a minimal scaffold.

Preserved:
- `dataset/`
- `results/`
- `hd-epic-annotations/`
- `thirdparty/`

Code areas to rebuild:
- `src/` for runtime system code
- `adapters/` for data readers and external input integration
- `eval/` for benchmark and evaluation code

Current runtime entrypoint:

```bash
PYTHONPATH=src .venv/bin/python main.py
```
