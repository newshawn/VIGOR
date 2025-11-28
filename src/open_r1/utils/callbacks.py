#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
from typing import List

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from .evaluation import run_benchmark_jobs
from .hub import push_to_hub_revision
from .wandb_logging import save_git_patch_if_possible


def is_slurm_available() -> bool:
    # returns true if a slurm queueing system is available
    try:
        subprocess.run(["sinfo"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


class DummyConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class PushToHubRevisionCallback(TrainerCallback):
    def __init__(self, model_config) -> None:
        self.model_config = model_config

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            global_step = state.global_step

            # WARNING: if you use dataclasses.replace(args, ...) the accelerator dist state will be broken, so I do this workaround
            # Also if you instantiate a new SFTConfig, the accelerator dist state will be broken
            dummy_config = DummyConfig(
                hub_model_id=args.hub_model_id,
                hub_model_revision=f"{args.hub_model_revision}-step-{global_step:09d}",
                output_dir=f"{args.output_dir}/checkpoint-{global_step}",
                system_prompt=args.system_prompt,
            )

            future = push_to_hub_revision(
                dummy_config, extra_ignore_patterns=["*.pt"]
            )  # don't push the optimizer states

            if is_slurm_available():
                dummy_config.benchmarks = args.benchmarks

                def run_benchmark_callback(_):
                    print(f"Checkpoint {global_step} pushed to hub.")
                    run_benchmark_jobs(dummy_config, self.model_config)

                future.add_done_callback(run_benchmark_callback)


class WandbGitPatchCallback(TrainerCallback):
    """
    Save the current git diff to the active wandb run once training starts.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        self._saved = False

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._saved or not state.is_world_process_zero:
            return
        save_git_patch_if_possible(getattr(args, "report_to", None))
        self._saved = True


def _wants_wandb_logging(train_config) -> bool:
    report_to = getattr(train_config, "report_to", None)
    if report_to is None:
        return False
    if isinstance(report_to, str):
        return "wandb" in report_to
    try:
        return "wandb" in list(report_to)
    except Exception:
        return False


def _flag_is_disabled(env_value: str) -> bool:
    return str(env_value).lower() in {"0", "false", "no", "off"}


CALLBACKS = {
    "push_to_hub_revision": PushToHubRevisionCallback,
    "wandb_git_patch": WandbGitPatchCallback,
}


def get_callbacks(train_config, model_config) -> List[TrainerCallback]:
    callbacks = []
    for callback_name in train_config.callbacks:
        if callback_name not in CALLBACKS:
            raise ValueError(f"Callback {callback_name} not found in CALLBACKS.")
        callbacks.append(CALLBACKS[callback_name](model_config))

    wandb_patch_enabled = not _flag_is_disabled(os.environ.get("INTUITOR_ENABLE_WANDB_GIT_PATCH", "1"))
    if wandb_patch_enabled and _wants_wandb_logging(train_config):
        already_added = any(isinstance(cb, WandbGitPatchCallback) for cb in callbacks)
        if not already_added:
            callbacks.append(WandbGitPatchCallback())

    return callbacks
