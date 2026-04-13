from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.experiments.hd_epic_mcq_shortlist_joint import BudgetConfig, run_direct_layer2_method


def run_method(
    video_id: str,
    tasks: list[str],
    limit: int,
    output_dir: Path,
    budget_config: BudgetConfig,
    answer_config: Any,
    *,
    answerer: Any | None = None,
) -> dict:
    return run_direct_layer2_method(
        video_id=video_id,
        tasks=tasks,
        limit=limit,
        output_dir=output_dir,
        budget_config=budget_config,
        answer_config=answer_config,
        answerer=answerer,
    )
