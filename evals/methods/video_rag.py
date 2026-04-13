from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.experiments.hd_epic_mcq_shortlist_joint import BudgetConfig


def run_method(
    video_id: str,
    tasks: list[str],
    limit: int,
    output_dir: Path,
    budget_config: BudgetConfig,
    answer_config: Any,
) -> dict:
    raise NotImplementedError("Deferred: not in first comparison scope")

__all__ = ["run_method"]
