import math
import os
import shutil
from typing import Any, Optional

import transformers
from packaging import version
from trl import GRPOTrainer


class GRPOTopKTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_top_k = max(0, int(getattr(self.args, "save_top_k", 0) or 0))
        self.save_top_k_metric = (getattr(self.args, "save_top_k_metric", None) or "").strip()
        self.save_top_k_greater_is_better = bool(getattr(self.args, "save_top_k_greater_is_better", True))
        self._top_k_checkpoints: list[dict[str, Any]] = []
        self._last_top_k_step = -1

    def _maybe_save_top_k_checkpoint(self, logs: dict[str, float]) -> None:
        if self.save_top_k <= 0:
            return
        if not self.save_top_k_metric:
            return
        if not self.is_world_process_zero():
            return
        metric_value = logs.get(self.save_top_k_metric)
        if metric_value is None and not self.save_top_k_metric.startswith("eval_"):
            metric_value = logs.get(f"eval_{self.save_top_k_metric}")
        if metric_value is None:
            return
        try:
            metric_value = float(metric_value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(metric_value):
            return
        step = int(self.state.global_step)
        if step <= 0 or step == self._last_top_k_step:
            return

        def is_better(new_metric: float, new_step: int, ref: dict[str, Any]) -> bool:
            if self.save_top_k_greater_is_better:
                return new_metric > ref["metric"] or (new_metric == ref["metric"] and new_step > ref["step"])
            return new_metric < ref["metric"] or (new_metric == ref["metric"] and new_step > ref["step"])

        def worst_checkpoint() -> dict[str, Any]:
            if self.save_top_k_greater_is_better:
                return min(self._top_k_checkpoints, key=lambda x: (x["metric"], x["step"]))
            return max(self._top_k_checkpoints, key=lambda x: (x["metric"], -x["step"]))

        if self._top_k_checkpoints:
            if len(self._top_k_checkpoints) >= self.save_top_k:
                worst = worst_checkpoint()
                if not is_better(metric_value, step, worst):
                    return

        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{step}")
        self.save_model(checkpoint_dir)
        self._top_k_checkpoints.append({"step": step, "metric": metric_value, "path": checkpoint_dir})
        self._last_top_k_step = step

        if len(self._top_k_checkpoints) > self.save_top_k:
            worst = worst_checkpoint()
            self._top_k_checkpoints.remove(worst)
            if os.path.isdir(worst["path"]):
                shutil.rmtree(worst["path"], ignore_errors=True)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        last_logs = self.state.log_history[-1] if getattr(self.state, "log_history", None) else logs
        self._maybe_save_top_k_checkpoint(last_logs)
