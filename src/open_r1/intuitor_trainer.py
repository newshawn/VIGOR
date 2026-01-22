# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

# This file is adapted from trl-0.18.0 grpo_trainer.py

import json
import math
import os
import shutil
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Sized
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Optional, Union

import datasets
import numpy as np
import torch
import torch.utils.data
import transformers
from accelerate.utils import (broadcast_object_list, gather, gather_object,
                              is_peft_model, set_seed)
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Sampler
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          GenerationConfig, PreTrainedModel,
                          PreTrainedTokenizerBase, Trainer, TrainerCallback,
                          is_wandb_available)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_callback import ExportableState

try:
    from transformers.trainer_utils import TRAINER_STATE_NAME
except ImportError:  # transformers<4.53
    from transformers.trainer import TRAINER_STATE_NAME

from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_peft_available
from trl.data_utils import (apply_chat_template, is_conversational,
                            maybe_apply_chat_template)
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import (is_liger_kernel_available, is_rich_available,
                              is_vllm_available)
from trl.models import (create_reference_model, prepare_deepspeed,
                        unwrap_model_for_generation)
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import (disable_dropout_in_model, generate_model_card,
                               get_comet_experiment_url, pad,
                               print_prompt_completions_sample,
                               selective_log_softmax)

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

if is_wandb_available():
    import wandb

try:
    from torch.utils.tensorboard import \
        SummaryWriter as _TorchSummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    _TorchSummaryWriter = None

try:
    from tensorboardX import SummaryWriter as _TBXSummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    _TBXSummaryWriter = None

_SummaryWriter = _TorchSummaryWriter or _TBXSummaryWriter

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class RepeatRandomSampler(RepeatSampler):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "RepeatRandomSampler is deprecated and will be removed in version 0.18. Use RepeatSampler instead.",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)


def normalize_per_prompt(values: torch.Tensor, group_size: int, eps: float = 1e-12) -> torch.Tensor:
    """
    Apply row-wise min-max normalization over flattened prompt groups.

    Args:
        values: Tensor shaped (num_prompts * group_size,)
        group_size: Number of completions per prompt.
    """
    if group_size <= 0 or values.numel() == 0 or values.numel() % group_size != 0:
        return values

    matrix = values.view(-1, group_size)
    row_min = matrix.min(dim=1, keepdim=True).values
    row_max = matrix.max(dim=1, keepdim=True).values
    row_span = row_max - row_min

    normalized = torch.zeros_like(matrix)
    stable_rows = (row_span >= eps).squeeze(1)
    if stable_rows.any():
        normalized_subset = (matrix[stable_rows] - row_min[stable_rows]) / row_span[stable_rows]
        normalized[stable_rows] = normalized_subset

    degenerate_rows = (~stable_rows).nonzero(as_tuple=True)[0]
    if degenerate_rows.numel() > 0:
        argmax_idx = matrix[degenerate_rows].argmax(dim=1)
        normalized[degenerate_rows, :] = 0.0
        normalized[degenerate_rows, argmax_idx] = 1.0

    return normalized.view_as(values)


def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
        >>> x = torch.arange(12).reshape(6, 2)
        >>> y = torch.arange(6).reshape(6, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> split_tensor_dict(tensor_dict, 3)
        [
            {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
            {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
            {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
        ]
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: tensor[i * chunk_size : (i + 1) * chunk_size] if tensor is not None else None
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


class INTUITORTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. Custom reward
                  functions can also return None when the reward is not applicable to those samples. This is useful for
                  multi-task training where different reward functions apply to different types of samples. When a
                  reward function returns None for a sample, that reward function is excluded from the reward
                  calculation for that sample. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`]. A
            padding token, `processing_class.pad_token`, must be set. If the processing class has not set a padding
            token, `processing_class.eos_token` will be used as the default.
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Optional[Union[RewardFunc, list[RewardFunc]]] = [],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        report_to = getattr(args, "report_to", None)
        if report_to == "all":
            os.environ.setdefault("WANDB_MODE", "offline")
        else:
            report_to_values = set()
            if isinstance(report_to, str):
                report_to_values.add(report_to)
            elif report_to:
                report_to_values.update(report_to)
            if "wandb" in report_to_values:
                os.environ.setdefault("WANDB_MODE", "offline")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            model = get_peft_model(model, peft_config)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        self.beta = args.beta
        self._needs_ref_model = self.beta != 0.0
        if not self._needs_ref_model:
            # Neither KL penalty nor reference reward is active; no ref model needed.
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif is_peft_model(model):
            # With PEFT we can disable adapters to recover the base weights instead of keeping a duplicate copy.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        else:
            processing_class.padding_side = "left"
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs[i], nn.Module):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names.append(reward_funcs[i].config._name_or_path.split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_vllm = args.use_vllm
        self.use_liger_loss = args.use_liger_loss
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.mask_truncated_completions = args.mask_truncated_completions
        kl_eta = getattr(args, "kl_reward_eta", None)
        if kl_eta is None:
            kl_eta = getattr(args, "learning_rate", None)
        if kl_eta is None:
            kl_eta = 1.0
        self.kl_reward_eta = float(kl_eta)
        self._kl_reward_params_cache = None
        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        if self.use_liger_loss:
            if not is_liger_kernel_available():
                raise ImportError(
                    "Liger is required to use `liger_loss` as the GRPO loss. Run `pip install liger-kernel`."
                )
            if is_peft_model(model):
                raise TypeError("Liger loss is not supported with a PEFT model.")

            if self.loss_type != "bnpo":
                raise ValueError(
                    f"The provided loss type (`{self.loss_type}`) is not supported with `use_liger_loss`. Liger loss "
                    "only supports `bnpo` for now."
                )

            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.ref_model is not None,
            )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.log_completions = False
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        self._wandb_artifact_logged = False
        self._tb_writer = None
        self._tb_log_dir: Optional[str] = None
        self._tb_warning_emitted = False
        self._maybe_init_tensorboard_writer()
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "entropy": deque(maxlen=maxlen),
            "logprob": deque(maxlen=maxlen),
            "p_norm": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
            # Aggregated reward (after applying weights) per completion
            "aggregated_reward": deque(maxlen=maxlen),
            # Advantages captured for logging (per completion)
            "advantages": {
                "reward": deque(maxlen=maxlen),   # from task rewards
                "kl": deque(maxlen=maxlen),       # from KL term
                "total": deque(maxlen=maxlen),    # reward + kl
            },
        }
        # Cursor for grouped textual logging
        self._last_grouped_log_index = 0
        # Buffers for optional KL reward plotting
        self.kl_reward_plot_enabled = bool(getattr(args, "kl_reward_plot_enabled", False))
        self.kl_reward_plot_every_n_steps = max(1, int(getattr(args, "kl_reward_plot_every_n_steps", 1)))
        self.kl_reward_plot_max_prompts = max(1, int(getattr(args, "kl_reward_plot_max_prompts", 30)))
        self._kl_plot_buffer = {
            "prompt": deque(maxlen=maxlen),
            "kl_reward_norm": deque(maxlen=maxlen),
            "completion_length": deque(maxlen=maxlen),
            "repeat_ngram3": deque(maxlen=maxlen),
            "accuracy_reward": deque(maxlen=maxlen),
        }
        self._kl_plot_dir = Path(self.args.output_dir).resolve() / "kl_reward_plots"
        self._last_kl_plot_step = -1
        self._kl_plot_warning_emitted = False
        self.prompt_dump_enabled = bool(getattr(args, "prompt_dump_enabled", False))
        self.prompt_dump_every_n_steps = max(1, int(getattr(args, "prompt_dump_every_n_steps", 5)))
        self.prompt_dump_max_prompts = max(1, int(getattr(args, "prompt_dump_max_prompts", 50)))
        self._prompt_dump_dir = (
            Path(self.args.output_dir).resolve() / "prompt_completions" if self.prompt_dump_enabled else None
        )
        self._last_prompt_dump_step = -1
        if self.prompt_dump_enabled and self.accelerator.is_main_process:
            self._prompt_dump_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = max(0, int(getattr(args, "save_top_k", 0) or 0))
        self.save_top_k_metric = (getattr(args, "save_top_k_metric", None) or "").strip()
        self.save_top_k_greater_is_better = bool(getattr(args, "save_top_k_greater_is_better", True))
        self._top_k_checkpoints: list[dict[str, Any]] = []
        self._last_top_k_step = -1
        self.save_resume_steps = max(0, int(getattr(args, "save_resume_steps", 0) or 0))
        self._last_resume_step = -1

        # Check if the effective batch size can be divided by the number of generations
        if self.num_generations < 2:
            raise ValueError(
                "GRPO requires at least 2 generations per prompt to calculate the advantages. You provided "
                f"{self.num_generations}, which is less than the minimum required."
            )
        num_processes = self.accelerator.num_processes
        effective_batch_size = args.per_device_train_batch_size * num_processes * args.gradient_accumulation_steps
        possible_values = [
            n_gen for n_gen in range(2, effective_batch_size + 1) if (effective_batch_size) % n_gen == 0
        ]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The effective train batch size ({num_processes} x {args.per_device_train_batch_size} x "
                f"{args.gradient_accumulation_steps}) must be evenly divisible by the number of generations per "
                f"prompt ({self.num_generations}). Given the current effective train batch size, the valid values for "
                f"the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            effective_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [
                n_gen for n_gen in range(2, effective_batch_size + 1) if (effective_batch_size) % n_gen == 0
            ]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The effective eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be "
                    f"evenly divisible by the number of generations per prompt ({self.num_generations}). Given the "
                    "current effective eval batch size, the valid values for the number of generations are: "
                    f"{possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # Log optional components to make run config explicit.
        # self._log_optional_modules()

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                self.vllm_client = VLLMClient(
                    args.vllm_server_host, args.vllm_server_port, connection_timeout=args.vllm_server_timeout
                )
                self.vllm_client.init_communicator()

            # vLLM specific sampling arguments
            self.guided_decoding_regex = args.vllm_guided_decoding_regex

            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                pad_token_id=processing_class.pad_token_id,
                bos_token_id=processing_class.bos_token_id,
                eos_token_id=processing_class.eos_token_id,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                repetition_penalty=self.repetition_penalty,
                cache_implementation=args.cache_implementation,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch, our dataloader loads an *accumulated* batch
    # (i.e., `per_device_batch_size × gradient_accumulation_steps`). This allows us to generate completions
    # once per optimization step—rather than once per gradient accumulation step—which is significantly more efficient.
    # The only change from the original implementation is multiplying the batch size by `gradient_accumulation_steps`.
    # Thus, `_prepare_inputs` is called with the accumulated batch size, and it handles the splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification.As a result, some parts of the method aren't relevant to GRPO, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.gradient_accumulation_steps,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |     Accum step 0      |     Accum step 1      |
        #                                      |   GPU 0   |   GPU 1   |   GPU 0   |   GPU 1   |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     [0   0   1   1   2   2]  3   3   4   4   5   5    <- Generate for the whole accumulated batch; store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1      0   0   1   1   2   2 [ 3   3   4   4   5   5]   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     [0   0   1   1   2   2]  3   3   4   4   5   5    <- Take the stored generations and use the first slice to compute the loss
        #  num_iterations=2 ▼  1          3      0   0   1   1   2   2 [ 3   3   4   4   5   5]   <- Take the stored generations and use the second slice to compute the loss
        #
        #                      2          4     [6   6   7   7   8   8]  9   9  10  10  11  11    <- Generate for the whole accumulated batch; store the completions; use the first slice to compute the loss
        #                      2          5      6   6   7   7   8   8 [ 9   9  10  10  11  11]   <- ...
        #                                          ...
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return RepeatSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.gradient_accumulation_steps,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    @profiling_decorator
    def _get_last_hidden_state(self, model, input_ids, attention_mask, logits_to_keep=None):
        # unwrap the model to access the model.model
        unwrapped_model = self.accelerator.unwrap_model(model)
        last_hidden_state = unwrapped_model.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]

            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                logits_to_keep=logits_to_keep + 1,
                use_cache=False,
            ).logits
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            logps = selective_log_softmax(logits, input_ids_batch)  # compute logprobs for the input tokens
            all_logps.append(logps)
            # import pdb; pdb.set_trace()
        return torch.cat(all_logps, dim=0)

    def _get_kl_reward_parameters(self, model: nn.Module) -> list[torch.nn.Parameter]:
        if getattr(self, "_kl_reward_params_cache", None) is not None:
            return self._kl_reward_params_cache

        unwrap_model = self.accelerator.unwrap_model(model)
        lora_params: list[torch.nn.Parameter] = []
        fallback_params: list[torch.nn.Parameter] = []
        for name, param in unwrap_model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora" in name.lower():
                lora_params.append(param)
            else:
                fallback_params.append(param)

        if lora_params:
            selected_params = lora_params
        else:
            selected_params = fallback_params

        self._kl_reward_params_cache = selected_params
        return self._kl_reward_params_cache

    @profiling_decorator
    def _compute_gradient_kl_reward(
        self,
        model: nn.Module,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        trainable_params = self._get_kl_reward_parameters(model)
        if len(trainable_params) == 0:
            zero = torch.zeros(prompt_ids.size(0), device=prompt_ids.device, dtype=torch.float32)
            return zero, zero, zero, zero

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        prompt_labels = torch.full_like(prompt_ids, -100)
        completion_labels = completion_ids.clone()
        completion_labels = completion_labels.masked_fill(completion_mask == 0, -100)
        labels = torch.cat([prompt_labels, completion_labels], dim=1)
        # eta 是 KL 奖励的缩放因子，通常设置为学习率
        eta = getattr(self, "kl_reward_eta", None)
        if eta is None:
            eta = getattr(self.args, "learning_rate", 1.0)
        eta_sq = float(eta) * float(eta)
        eta_sq = torch.tensor(eta_sq, device=prompt_ids.device, dtype=torch.float32)

        rewards: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        logprobs: list[torch.Tensor] = []
        p_norms: list[torch.Tensor] = []
        truncated_flags: list[bool] = []
        model_device = prompt_ids.device
        for idx in range(input_ids.size(0)):
            model.zero_grad(set_to_none=True)
            single_input_ids = input_ids[idx : idx + 1]
            single_attention_mask = attention_mask[idx : idx + 1]
            single_labels = labels[idx : idx + 1]

            if completion_mask[idx].sum() <= 0:
                truncated_flags.append(True)
                nan_scalar = torch.tensor(float("nan"), device=model_device, dtype=torch.float32)
                rewards.append(nan_scalar)
                entropies.append(nan_scalar)
                logprobs.append(nan_scalar)
                p_norms.append(nan_scalar)
                continue
            truncated_flags.append(False)

            with torch.enable_grad():
                # Use bf16 autocast to reduce activation memory during KL gradient computation
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=single_input_ids,
                        attention_mask=single_attention_mask,
                        labels=single_labels,
                        use_cache=False,
                    )
                    loss = outputs.loss
                loss = loss.float()
                grads = torch.autograd.grad(
                    loss,
                    trainable_params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
            grad_norm = torch.sqrt(sum([(g**2).sum() for g in grads if g is not None]))
            # grad_norm_sq = torch.zeros(1, device=model_device, dtype=torch.float32)
            # for grad in grads:
            #     if grad is None:
            #         continue
            #     grad_norm_sq += grad.float().pow(2).sum()

            # approx_kl = 0.5 * eta_sq * grad_norm_sq
            token_mask = single_labels[:, 1:].ne(-100)  # 对齐到 logits[:, :-1, :]
            if getattr(self.args, "kl_reward_sqrt_len_scaling_enabled", True):
                eff_len = token_mask.sum().to(device=model_device, dtype=grad_norm.dtype)
                grad_norm = grad_norm * torch.sqrt(eff_len.clamp_min(1))
            approx_kl = grad_norm
            rewards.append((-approx_kl).squeeze(0))
            # import pdb; pdb.set_trace()
            # 提前 detach logits 并释放 outputs，减少显存滞留
            logits_detached = outputs.logits.detach() if outputs.logits is not None else None
            del outputs

            # 计算 completion 的平均熵（只统计标签为有效 token 的位置）
            
            comp_len = int(token_mask.sum().item())
            # 为避免保留计算图，熵计算在 no_grad 下进行，并先 detach logits
            if comp_len > 0 and logits_detached is not None:
                with torch.no_grad():
                    logits = logits_detached[:, :-1, :].float()  # align to target tokens
                    log_probs = logits.log_softmax(dim=-1)  # [1, seq_len-1, vocab_size]
                    token_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)  # (1, seq_len-1)
                    mask = token_mask.float()
                    avg_entropy = (token_entropy * mask).sum() / comp_len
            else:
                avg_entropy = torch.zeros(1, device=model_device, dtype=torch.float32).squeeze(0)
            entropies.append(avg_entropy)

            # 计算对数似然 log p(completion | prompt) = sum log p(c_i | prompt, c_<i)
            labels_shifted = single_labels[:, 1:]  # 对齐 logits[:, :-1, :]
            with torch.no_grad():
                if logits_detached is not None:
                    log_probs_all = logits_detached[:, :-1, :].float().log_softmax(dim=-1)
                    gather_labels = labels_shifted.masked_fill(token_mask == 0, 0).unsqueeze(-1)  # 避免非法索引
                    token_log_probs = torch.gather(log_probs_all, dim=-1, index=gather_labels).squeeze(-1)
                    seq_log_prob = (token_log_probs * token_mask.float()).sum()
                    if comp_len > 0:
                        token_probs = token_log_probs.exp()
                        mean_token_prob = (token_probs * token_mask.float()).sum() / float(comp_len)
                    else:
                        mean_token_prob = torch.zeros(1, device=model_device, dtype=torch.float32).squeeze(0)
                else:
                    seq_log_prob = torch.zeros(1, device=model_device, dtype=torch.float32).squeeze(0)
                    mean_token_prob = torch.zeros(1, device=model_device, dtype=torch.float32).squeeze(0)
            logprobs.append(seq_log_prob)
            # 归一化后的平均 token 概率（算术均值）
            p_norms.append(mean_token_prob)
            # 释放局部变量，减少显存滞留
            del loss, grads, logits_detached

        rewards_tensor = torch.stack(rewards).to(device=model_device, dtype=torch.float32)
        entropies_tensor = torch.stack(entropies).to(device=model_device, dtype=torch.float32)
        logprobs_tensor = torch.stack(logprobs).to(device=model_device, dtype=torch.float32)
        p_norms_tensor = torch.stack(p_norms).to(device=model_device, dtype=torch.float32)

        # 将截断样本替换为“批内最差”值，确保排序时始终垫底
        # 可以把margin都设置为1e-3，然后-||g||比最差的低margin，p_norm比最高的高margin，entropy比最低的低margin，logprobs比最低的低margin
        truncated_mask = torch.tensor(truncated_flags, device=model_device, dtype=torch.bool)
        if truncated_mask.any().item():
            valid_mask = ~truncated_mask
            if valid_mask.any().item():
                margin = 1e-3
                fallback_reward = rewards_tensor[valid_mask].min() - margin
                fallback_entropy = entropies_tensor[valid_mask].min() - margin
                fallback_logprob = logprobs_tensor[valid_mask].min() - margin
                fallback_p_norm = p_norms_tensor[valid_mask].max() + margin
            else:
                # 极端场景：全是截断样本，退回固定兜底值
                fallback_reward = torch.tensor(-20.0, device=model_device, dtype=torch.float32)
                fallback_entropy = torch.tensor(0.1, device=model_device, dtype=torch.float32)
                fallback_logprob = torch.tensor(0.0, device=model_device, dtype=torch.float32)
                fallback_p_norm = torch.tensor(0.95, device=model_device, dtype=torch.float32)

            rewards_tensor = torch.where(truncated_mask, fallback_reward, rewards_tensor)
            entropies_tensor = torch.where(truncated_mask, fallback_entropy, entropies_tensor)
            logprobs_tensor = torch.where(truncated_mask, fallback_logprob, logprobs_tensor)
            p_norms_tensor = torch.where(truncated_mask, fallback_p_norm, p_norms_tensor)

        return (
            rewards_tensor,
            entropies_tensor,
            logprobs_tensor,
            p_norms_tensor,
        )



    # def _log_optional_modules(self) -> None:
    #     """Log which optional components are enabled/disabled for transparency."""
    #     log_fn = getattr(self.accelerator, "print", print)
    #     log_fn("[IntuitorTrainer] no optional ref_reward enabled")


    @profiling_decorator
    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model):
            # With PEFT and DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as merging
            # adapters in a sharded manner is not supported.
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                for name, param in self.model.named_parameters():
                    # When using PEFT, we need to recover the original parameter name and discard some parameters
                    name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                    if self.model.prefix in name:
                        continue
                    # When module to save, remove its prefix and discard the original module
                    if "original_module" in name:
                        continue
                    name = name.replace("modules_to_save.default.", "")

                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)

                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather and update each parameter individually.
            for name, param in self.model.named_parameters():
                with gather_if_zero3([param]):
                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)

        # Reset cache on main process
        if self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()

    @profiling_decorator
    def _prepare_inputs(
        self, accumulated_local_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the accumulated local batch (Per-GPU batch size × Gradient accumulation steps)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire accumulated batch and splits it into smaller batches
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every gradient_accumulation_steps * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.gradient_accumulation_steps * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                accumulated_local_batch = self._generate_and_score_completions(accumulated_local_batch)
                self._buffered_inputs = split_tensor_dict(
                    accumulated_local_batch, self.args.gradient_accumulation_steps
                )
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, there is neither gradient accumulation, nor multiple iterations
            inputs = self._generate_and_score_completions(accumulated_local_batch)
        return inputs

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.
                ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                with profiling_context(self, "vLLM.generate"):
                    completion_ids = self.vllm_client.generate(
                        prompts=ordered_set_of_prompts,
                        n=self.num_generations,
                        repetition_penalty=self.repetition_penalty,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=-1 if self.top_k is None else self.top_k,
                        min_p=0.0 if self.min_p is None else self.min_p,
                        max_tokens=self.max_completion_length,
                        guided_decoding_regex=self.guided_decoding_regex,
                    )
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        # 样本1: [10, 20, 2, 30, 40]  # EOS在位置2
        # 样本2: [15, 25, 35, 45]     # 没有EOS
        # is_eos: [[False, False, True, False, False],
        #         [False, False, False, False]]
        # eos_idx: [2, 4]  # 样本1的EOS在位置2，样本2没有EOS（设为序列长度4）
        # completion_mask: [[1, 1, 1, 0, 0],  # 位置2（EOS）及之后被屏蔽
        #                 [1, 1, 1, 1]]     # 没有EOS，全部保留
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            # eos_idx现在包含每个样本第一个EOS token的位置，没有EOS的样本保持序列长度
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        # 使用截断，例如一个句子没有EOS，那么就说明这个句子没有说完，拿来训练效果肯定不好，就将其掩码都设置为0
            # 样本1: [10, 20, 2, 30]    # 正常结束（EOS在位置2）
            # 样本2: [15, 25, 35]       # 被截断（无EOS）
            # 样本3: [12, 22, 2, 32]    # 正常结束（EOS在位置2）
            # is_eos.any(dim=1) = [True, False, True]  # 样本2被截断
            # truncated_completions = [False, True, False]  # 样本2被标记为截断
            # (~truncated_completions) = [True, False, True]  # 取反
            # 扩展后: [[1], [0], [1]]  # unsqueeze(1)结果
            # 最终completion_mask: 原掩码 * [[1,1,1,1], [0,0,0], [1,1,1,1]]
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        # 计算kl奖励，并返回每个 completion 的平均熵/对数似然（后续用于日志/指标）
        neg_grad_L2, completion_entropies, completion_logps, completion_p_norms = self._compute_gradient_kl_reward(
            self.model, prompt_ids, prompt_mask, completion_ids, completion_mask
        )
        neg_grad_L2 = neg_grad_L2.detach()
        completion_entropies = completion_entropies.detach()
        completion_logps = completion_logps.detach()
        completion_p_norms = completion_p_norms.detach()
        kl_rewards = neg_grad_L2
        kl_rewards = kl_rewards.detach()
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:   # 只迭代一次，不需要保存旧策略
                old_per_token_logps = None

            if not self._needs_ref_model:
                ref_per_token_logps = None
            elif self.ref_model is not None:    # not None
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # === 逐个 reward func 打分：统一收集模块化模型输出与自定义 callable 返回值，reward_func=accuracy 但是 weights=0 ===
        rewards_per_func = torch.zeros(len(prompts), max(len(self.reward_funcs),1), device=device)
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # === 多卡同步：收集各类奖励/掩码，后续统一做归一化与日志记录 ===
        rewards_per_func = gather(rewards_per_func)
        gathered_neg_grad_L2 = gather(neg_grad_L2).reshape(-1)    # 将所有gpu的 -||g|| 聚集到一起，例如单卡shape=[6]，那么gathered_neg_grad_L2.shape=[12]
        gathered_kl_rewards = gather(kl_rewards).reshape(-1)
        gathered_entropies = gather(completion_entropies).reshape(-1)
        gathered_logprobs = gather(completion_logps).reshape(-1)
        gathered_p_norms = gather(completion_p_norms).reshape(-1)
        # === 目的是得到 -||g||（未加熵惩罚）的 mean/std 记录到 wandb 上（不再使用多样性 bonus） ===
        # 因为 std 需要每个 prompt 组内计算 std，再取平均值，不能所有的 prompt 的组放一起取平均值
        neg_grad_L2_by_prompt = gathered_neg_grad_L2.view(-1, self.num_generations)
        mean_neg_grad_L2 = neg_grad_L2_by_prompt.mean(dim=1)
        std_neg_grad_L2 = neg_grad_L2_by_prompt.std(dim=1)
        neg_grad_L2_record = mean_neg_grad_L2.mean().item()
        neg_grad_L2_std_record = std_neg_grad_L2.mean().item()
        entropy_weight_mean_record = float("nan")
        entropy_weight_std_record = float("nan")
        focal_lambda_record = float("nan")

        def _mean_and_std(tensor: torch.Tensor) -> tuple[float, float]:
            if tensor.numel() == 0:
                return float("nan"), float("nan")
            return float(tensor.mean().item()), float(tensor.std(unbiased=False).item())

        entropy_mean_record, entropy_std_record = _mean_and_std(gathered_entropies)
        logprob_mean_record, logprob_std_record = _mean_and_std(gathered_logprobs)
        pnorm_mean_record, pnorm_std_record = _mean_and_std(gathered_p_norms)
        kl_rewards_by_prompt = gathered_kl_rewards.view(-1, self.num_generations)

        # === 按 completion 熵做 focal 风格的 KL 奖励缩放：Reward = -||g|| * (H_avg)^lambda 或者 Reward = -||g|| * (1 - p_norm)^lambda ===
        if getattr(self.args, "kl_entropy_weighting_enabled", False) and gathered_entropies.numel() > 0:
            entropy_eps = 1e-6
            # 固定 lambda，不再使用 warmup/线性/余弦衰减
            focal_lambda = float(getattr(self.args, "kl_entropy_focal_lambda", -1))
            focal_metric = str(getattr(self.args, "kl_entropy_focal_metric", "entropy")).lower()
            if focal_metric == "p_norm":
                # 仅对超出目标 p_norm 的 completion 施加惩罚，并在极端置信度时做裁剪避免梯度崩塌
                p_norm_vals = gathered_p_norms.view(-1, self.num_generations)
                pnorm_target = getattr(self.args, "kl_entropy_pnorm_target", 0.9)
                base = torch.ones_like(p_norm_vals)
                if pnorm_target is not None:
                    pnorm_mask = p_norm_vals > float(pnorm_target)
                    base = torch.where(
                        pnorm_mask,
                        (1.0 - p_norm_vals).clamp_min(entropy_eps),  # 高置信度才用小数
                        torch.ones_like(p_norm_vals),                # 否则就用1，这样1^lambda=1，不改变奖励
                    )
            else:
                entropy_target = getattr(self.args, "kl_entropy_low_entropy_target", None)
                base = (gathered_entropies.view(-1, self.num_generations) + entropy_eps).clamp_min(entropy_eps)
                # import pdb; pdb.set_trace()
                if entropy_target is not None:
                    entropy_mask = base < float(entropy_target)
                    # 高于阈值的直接用 1，不再额外 mask 一次
                    base = torch.where(entropy_mask, base, torch.ones_like(base))
            # import pdb; pdb.set_trace()
            focal_weights = base.pow(focal_lambda)
            kl_rewards_by_prompt = kl_rewards_by_prompt * focal_weights
            entropy_weight_mean_record = focal_weights.mean().item()
            entropy_weight_std_record = focal_weights.std(unbiased=False).item()
            focal_lambda_record = float(focal_lambda)
            # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # === KL 奖励归一：按每个 prompt 内（可含熵惩罚的）kl_rewards 做 rank，映射到 [-1, 1] ===
        # kl_rewards_by_prompt 为负数，绝对值越小（越靠近 0）表示梯度越小 → 奖励越高
        # 按值升序排序：最负的 rank 最低，最接近 0 的 rank 最高
        if getattr(self.args, "kl_reward_rank_normalization_enabled", True):
            rank_idx = kl_rewards_by_prompt.argsort(dim=1, stable=True)
            ranks = rank_idx.argsort(dim=1, stable=True).float()  # [num_prompt, num_generations] 数值越大越“好”
            if self.num_generations > 1:
                ranks = (ranks / (self.num_generations - 1)) * 2.0 - 1.0
            else:
                ranks = torch.zeros_like(ranks)
            kl_rewards_norm_flat = ranks.reshape(-1)  # shape=[num_prompt*num_generations*num_device]
        else:
            kl_rewards_norm_flat = kl_rewards_by_prompt.reshape(-1)
        gathered_kl_rewards = kl_rewards_norm_flat

        kl_rewards_norm_for_logging = kl_rewards_norm_flat.detach().cpu().tolist()  # 保存 KL 奖励用于文本日志
        # === 任务奖励聚合与优势计算（这里只有accuracy_reward，weight=0，不用于优化） ===    
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)   # Compute grouped-wise rewards
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        if torch.any(self.reward_weights != 0) and len(self.reward_funcs) > 0:  # if reward_weights is not all zero, normalize them to get the advantages, otherwise set advantage to 0
            warnings.warn("Combining specified rewards together with KL-based advantage.")
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = rewards - mean_grouped_rewards
            if self.scale_rewards:
                advantages = advantages / (std_grouped_rewards + 1e-4)
        else:
            advantages = torch.zeros_like(rewards)
        # if self.accelerator.process_index == 0:  # 或其它 rank
        #     import pdb; pdb.set_trace()
        # === 只保留当前 rank 对应的数据片段 ===
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        # === 归一化之后的 KL rewards：将gathered_kl_rewards按照prompt划分，计算组内mean和std，便于计算优势 ===
        gathered_kl_rewards = gathered_kl_rewards.view(-1, self.num_generations)    # (num_prompt, num_generations)
        mean_kl_rewards = gathered_kl_rewards.mean(dim=1)   # (num_prompt) 每个 prompt 小组内部做均值
        std_kl_rewards = gathered_kl_rewards.std(dim=1)

        kl_reward_record = mean_kl_rewards.mean().item()    # 计算所有 prompt 组的平均 KL 奖励 shape=[1]
        kl_reward_std_record = std_kl_rewards.mean().item()    # 计算所有 prompt 组的平均 KL 奖励标准差
        mean_kl_rewards = mean_kl_rewards.repeat_interleave(self.num_generations, dim=0) # 会将每个元素重复 num_generations 次（num_prompt*num_generations）。如果 mean_kl_rewards = [0.5, 0.3, 0.8] 且 num_generations = 4，结果将是 [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.8, 0.8, 0.8, 0.8]
        std_kl_rewards = std_kl_rewards.repeat_interleave(self.num_generations, dim=0)


        # === process_slice 获取当前 gpu 上的样本索引范围，计算当前 gpu 的 advantages ===
        kl_rewards = kl_rewards_norm_flat[process_slice]    # kl_rewards_norm_flat是归一化之后的shape=[num_prompt*num_generation]，使用[process_slice]切出当前gpu的部分，用于计算优势
        mean_kl_rewards = mean_kl_rewards[process_slice]
        std_kl_rewards = std_kl_rewards[process_slice]
        kl_advantage = (kl_rewards - mean_kl_rewards) / (std_kl_rewards + 1e-4)        # 逐个元素相减，计算 KL 奖励的优势
        total_advantage = kl_advantage + advantages
        # if self.accelerator.process_index == 0:  # 或其它 rank
        #     import pdb; pdb.set_trace()
        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # log completion lengths, mean, min, max
        agg_completion_mask = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_mask.float().max().item())
        # repetition collapse diagnostics using n-gram repetition ratio (higher means more repetition)
        def _ngram_repetition_ratio(token_ids, n=3):
            if len(token_ids) < n:
                return 0.0
            ngrams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
            return 1.0 - (len(set(ngrams)) / len(ngrams))

        completion_ids_cpu = completion_ids.detach().cpu()
        completion_mask_cpu = completion_mask.detach().cpu().bool()
        rep_ratios = []
        for ids, mask in zip(completion_ids_cpu, completion_mask_cpu):
            token_ids = ids[mask].tolist()
            rep_ratios.append(_ngram_repetition_ratio(token_ids, n=3))
        rep_ratios = torch.tensor(rep_ratios, device=device)
        agg_rep_ratios = self.accelerator.gather_for_metrics(rep_ratios)
        self._metrics[mode]["completions/repeat_ngram3_mean"].append(agg_rep_ratios.mean().item())
        self._metrics[mode]["completions/repeat_ngram3_max"].append(agg_rep_ratios.max().item())
        self._metrics[mode]["completions/repeat_ngram3_collapse_rate"].append(
            (agg_rep_ratios >= 0.5).float().mean().item()
        )

        # identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_mask) == 0:
            # edge case where no completed sequences are found
            term_completion_mask = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_mask.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["rewards/kl_reward/mean"].append(kl_reward_record)
        self._metrics[mode]["rewards/kl_reward/std"].append(kl_reward_std_record)
        self._metrics[mode]["rewards/neg_grad_L2/mean"].append(neg_grad_L2_record)
        self._metrics[mode]["rewards/neg_grad_L2/std"].append(neg_grad_L2_std_record)
        self._metrics[mode]["rewards/kl_reward_raw/mean"].append(neg_grad_L2_record)
        self._metrics[mode]["rewards/kl_reward_raw/std"].append(neg_grad_L2_std_record)
        self._metrics[mode]["entropy/mean"].append(entropy_mean_record)
        self._metrics[mode]["entropy/std"].append(entropy_std_record)
        self._metrics[mode]["logprob/mean"].append(logprob_mean_record)
        self._metrics[mode]["logprob/std"].append(logprob_std_record)
        self._metrics[mode]["p_norm/mean"].append(pnorm_mean_record)
        self._metrics[mode]["p_norm/std"].append(pnorm_std_record)
        self._metrics[mode]["rewards/kl_entropy_weight/mean"].append(entropy_weight_mean_record)
        self._metrics[mode]["rewards/kl_entropy_weight/std"].append(entropy_weight_std_record)
        self._metrics[mode]["rewards/kl_entropy_focal_lambda"].append(focal_lambda_record)
        # Alignment diagnostics: accuracy on high self-certainty subset vs overall.
        # Use kl_rewards_norm_flat as a proxy for SCE when selecting top-25% within each prompt group.
        acc_idx = next((i for i, name in enumerate(self.reward_func_names) if "accuracy" in name), None)
        if acc_idx is not None:
            acc = rewards_per_func[:, acc_idx]
            valid_mask = ~torch.isnan(acc)
            if valid_mask.any():
                acc_valid = acc[valid_mask]
                self._metrics[mode]["align/acc_overall"].append(acc_valid.float().mean().item())
                acc_grouped = acc.detach().cpu().view(-1, self.num_generations)
                kl_grouped = kl_rewards_norm_flat.detach().cpu().view(-1, self.num_generations)
                valid_grouped = valid_mask.detach().cpu().view(-1, self.num_generations)
                high_acc_values = []
                for acc_row, kl_row, valid_row in zip(acc_grouped, kl_grouped, valid_grouped):
                    valid_idx = valid_row.nonzero(as_tuple=False).squeeze(-1)
                    if valid_idx.numel() == 0:
                        continue
                    num_valid = valid_idx.numel()
                    top_k = max(1, (num_valid + 3) // 4)  # ceil(0.25 * num_valid)
                    kl_valid = kl_row[valid_idx]
                    _, topk_pos = torch.topk(kl_valid, top_k)
                    acc_top = acc_row[valid_idx[topk_pos]]
                    high_acc_values.append(acc_top)
                if high_acc_values:
                    acc_high = torch.cat(high_acc_values)
                    self._metrics[mode]["align/acc_high_sce_top25"].append(acc_high.float().mean().item())
                    self._metrics[mode]["align/wrong_high_sce_rate_top25"].append(
                        (acc_high <= 0).float().mean().item()
                    )
        # 监测 kl 的 advantage的平均值，为0才正常
        self._metrics[mode]["advantages/kl_mean_advantage"].append(self.accelerator.gather_for_metrics(kl_advantage).nanmean().item())
        self._metrics[mode]["advantages/total_mean_advantage"].append(self.accelerator.gather_for_metrics(total_advantage).nanmean().item())
        # Log prompt and completion texts
        gathered_prompts = gather_object(prompts_text)
        gathered_completions = gather_object(completions_text)
        if gathered_prompts and isinstance(gathered_prompts[0], (list, tuple)):
            gathered_prompts = [prompt for group in gathered_prompts for prompt in group]
        if gathered_completions and isinstance(gathered_completions[0], (list, tuple)):
            gathered_completions = [completion for group in gathered_completions for completion in group]
        self._textual_logs["prompt"].extend(gathered_prompts)
        self._textual_logs["completion"].extend(gathered_completions)
        self._textual_logs["entropy"].extend(gathered_entropies.detach().cpu().tolist())
        self._textual_logs["logprob"].extend(gathered_logprobs.detach().cpu().tolist())
        self._textual_logs["p_norm"].extend(gathered_p_norms.detach().cpu().tolist())
        if mode == "train" and self.kl_reward_plot_enabled:
            self._kl_plot_buffer["prompt"].extend(gathered_prompts)
            self._kl_plot_buffer["kl_reward_norm"].extend(kl_rewards_norm_for_logging)
            self._kl_plot_buffer["completion_length"].extend(agg_completion_mask.detach().cpu().tolist())
            self._kl_plot_buffer["repeat_ngram3"].extend(agg_rep_ratios.detach().cpu().tolist())
            if acc_idx is not None:
                acc_values_for_plot = rewards_per_func[:, acc_idx].detach().cpu().tolist()
            else:
                acc_values_for_plot = [float("nan")] * len(agg_completion_mask)
            self._kl_plot_buffer["accuracy_reward"].extend(acc_values_for_plot)
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())


        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": total_advantage,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }


    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.use_liger_loss:
            # Compute the loss using the liger grpo loss
            return self.compute_liger_loss(model, inputs)
        else:
            return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]   # (B, prompt_len)
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]   
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, seq_len)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        # (batch_size, sequence_length) 表明每个 completion token 的概率值的对数
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)  # (B, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            # KL 散度的一阶泰勒近似，这里是逐token计算的，相当于上面的B * logits_to_keep 个 token 分别计算 KL 散度
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"].unsqueeze(1)  # (B, 1) 句子级别的优势
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        # 重要性采样比率，per_token_logps已经取过对数了，因此前面要用torch.exp()， coef_1 = π_θ(a|s) / π_θ_old(a|s) = exp(log π_θ(a|s) - log π_θ_old(a|s))
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        # 一个completion的所有token共用优势，但是系数不同，有高有低，取决于重要性采样和clip，现在coef_1=coef_2=2，因为每次更新只使用一次重要性采样
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)   # 使用最小值，advantages > 0时，避免过于激进；<0时限制惩罚
        # import pdb; pdb.set_trace()
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        # import pdb; pdb.set_trace()
        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()

        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            # import pdb; pdb.set_trace()
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def _save_resume_checkpoint(self, model, trial) -> None:
        checkpoint_dir = os.path.join(self._get_output_dir(trial=trial), "checkpoint-last")
        if self.args.should_save and os.path.isdir(checkpoint_dir):
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        if hasattr(self, "accelerator"):
            self.accelerator.wait_for_everyone()

        self.save_model(checkpoint_dir, _internal_call=True)
        self._save_optimizer_and_scheduler(checkpoint_dir)
        self._save_scaler(checkpoint_dir)
        self._save_rng_state(checkpoint_dir)

        if self.args.should_save:
            for cb in [
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.state.stateful_callbacks[cb_name], list):
                    self.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.state.stateful_callbacks[cb_name] = cb_state
            self.state.save_to_json(os.path.join(checkpoint_dir, TRAINER_STATE_NAME))

    def _maybe_save_resume_checkpoint(self, model, trial) -> None:
        if self.save_resume_steps <= 0:
            return
        step = int(self.state.global_step)
        if step <= 0 or step == self._last_resume_step:
            return
        if step % self.save_resume_steps != 0:
            return
        self._save_resume_checkpoint(model, trial)
        self._last_resume_step = step

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

    def _has_tensorboard_callback(self) -> bool:
        handler = getattr(self, "callback_handler", None)
        callbacks = getattr(handler, "callbacks", []) if handler is not None else []
        return any(cb.__class__.__name__ == "TensorBoardCallback" for cb in callbacks)

    def _maybe_init_tensorboard_writer(self) -> None:
        if self._tb_writer is not None:
            return
        if not self.accelerator.is_main_process:
            return
        # Avoid double-logging when users already enabled the Transformers TensorBoard integration.
        if self._has_tensorboard_callback():
            return
        if _SummaryWriter is None:
            if not self._tb_warning_emitted:
                self.accelerator.print(
                    "[TensorBoard] Disabled because 'tensorboard' is not installed. "
                    "Install it with `pip install tensorboard`."
                )
                self._tb_warning_emitted = True
            return

        log_dir = getattr(self.args, "logging_dir", None) or os.path.join(self.args.output_dir, "runs")
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self._tb_writer = _SummaryWriter(log_dir=str(log_dir))
            self._tb_log_dir = str(log_dir)
            self.accelerator.print(f"[TensorBoard] Writing scalars to {self._tb_log_dir}")
        except Exception as exc:  # pragma: no cover
            self._tb_writer = None
            self._tb_log_dir = None
            self.accelerator.print(f"[TensorBoard] Failed to initialize SummaryWriter: {exc}")

    @staticmethod
    def _coerce_to_scalar(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float, np.number)):
            return float(value)
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return float(value.detach().cpu().item())
        if isinstance(value, np.ndarray):
            if value.size != 1:
                return None
            return float(np.asarray(value).reshape(()).item())
        return None

    def _maybe_log_tensorboard_scalars(self, logs: dict[str, Any]) -> None:
        if not self.accelerator.is_main_process:
            return
        writer = self._tb_writer
        if writer is None:
            return
        step = int(getattr(self.state, "global_step", 0) or 0)
        for key, value in logs.items():
            scalar = self._coerce_to_scalar(value)
            if scalar is None:
                continue
            writer.add_scalar(str(key), scalar, step)
        writer.flush()

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        super()._maybe_log_save_evaluate(
            tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
            learning_rate=learning_rate,
        )
        self._maybe_save_resume_checkpoint(model, trial)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._maybe_log_tensorboard_scalars(logs)
        self._metrics[mode].clear()
        self._maybe_save_top_k_checkpoint(logs)

        # 在 wandb run 初始化后尽早上传代码快照，且只上传一次
        if not self._wandb_artifact_logged:
            self._maybe_log_wandb_artifacts()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._textual_logs["prompt"],
                    self._textual_logs["completion"],
                    self._textual_logs["rewards"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                    "entropy": self._textual_logs.get("entropy", []),
                    "logprob": self._textual_logs.get("logprob", []),
                    "p_norm": self._textual_logs.get("p_norm", []),
                    **self._textual_logs["rewards"],
                }
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})

        self._maybe_dump_prompt_completions()
        self._maybe_save_kl_reward_plot()

    def _maybe_dump_prompt_completions(self) -> None:
        """Persist prompt/completion samples to JSON snapshots at a fixed cadence."""
        if not self.prompt_dump_enabled or self._prompt_dump_dir is None:
            return
        if not self.accelerator.is_main_process:
            return
        step = int(self.state.global_step)
        if step <= 0 or step == self._last_prompt_dump_step:
            return
        if step % self.prompt_dump_every_n_steps != 0:
            return

        prompts = list(self._textual_logs["prompt"])
        completions = list(self._textual_logs["completion"])
        if not prompts or not completions:
            return
        sample_count = min(len(prompts), len(completions))
        reward_logs = {name: list(values) for name, values in self._textual_logs["rewards"].items()}
        entropy_logs = list(self._textual_logs.get("entropy", []))
        logprob_logs = list(self._textual_logs.get("logprob", []))
        pnorm_logs = list(self._textual_logs.get("p_norm", []))

        grouped: list[dict[str, Any]] = []
        prompt_to_index: dict[str, int] = {}
        for idx in range(sample_count):
            prompt = prompts[idx]
            completion = completions[idx]
            if prompt not in prompt_to_index:
                if len(grouped) >= self.prompt_dump_max_prompts:
                    continue
                prompt_to_index[prompt] = len(grouped)
                grouped.append({"prompt": prompt, "samples": []})
            entry = grouped[prompt_to_index[prompt]]
            sample: dict[str, Any] = {"completion": completion}
            if reward_logs:
                sample["rewards"] = {
                    name: reward_logs[name][idx]
                    for name in reward_logs
                    if idx < len(reward_logs[name])
                }
            if idx < len(entropy_logs):
                sample["entropy"] = entropy_logs[idx]
            if idx < len(logprob_logs):
                sample["logprob"] = logprob_logs[idx]
            if idx < len(pnorm_logs):
                sample["p_norm"] = pnorm_logs[idx]
            entry["samples"].append(sample)

        if not grouped:
            return

        step_dir = self._prompt_dump_dir / f"step-{step:06d}"
        if step_dir.exists():
            shutil.rmtree(step_dir)
        step_dir.mkdir(parents=True, exist_ok=True)
        for prompt_idx, payload in enumerate(grouped, start=1):
            file_path = step_dir / f"prompt-{prompt_idx:04d}.json"
            data = {"step": step, **payload}
            with file_path.open("w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False, indent=2)

        self._last_prompt_dump_step = step

    def _clear_reward_plot_buffer(self) -> None:
        for buf in self._kl_plot_buffer.values():
            buf.clear()

    def _maybe_save_kl_reward_plot(self) -> None:
        """Persist reward heatmaps whenever the feature is enabled."""
        if not self.kl_reward_plot_enabled:
            self._clear_reward_plot_buffer()
            return
        if not self.accelerator.is_main_process:
            self._clear_reward_plot_buffer()
            return
        if not self._kl_plot_buffer["prompt"]:
            return

        step = int(self.state.global_step)
        if step <= 0 or step == self._last_kl_plot_step:
            self._clear_reward_plot_buffer()
            return
        if step % self.kl_reward_plot_every_n_steps != 0:
            self._clear_reward_plot_buffer()
            self._last_kl_plot_step = step
            return

        prompts = list(self._kl_plot_buffer["prompt"])
        kl_rewards_norm = list(self._kl_plot_buffer["kl_reward_norm"])
        completion_lengths = list(self._kl_plot_buffer.get("completion_length", []))
        repeat_ratios = list(self._kl_plot_buffer.get("repeat_ngram3", []))
        acc_values = list(self._kl_plot_buffer.get("accuracy_reward", []))
        try:
            self._maybe_log_sce_bin_summary(
                step,
                prompts,
                kl_rewards_norm,
                completion_lengths,
                repeat_ratios,
                acc_values,
            )
            if plt is None:
                if not self._kl_plot_warning_emitted:
                    warnings.warn(
                        "matplotlib is not available in this environment; disabling reward plotting."
                    )
                    self._kl_plot_warning_emitted = True
            elif kl_rewards_norm:
                if len(prompts) != len(kl_rewards_norm):
                    warnings.warn("Prompt/KL buffer size mismatch; skipping KL reward plot for this step.")
                else:
                    self._save_reward_heatmap(
                        step,
                        prompts,
                        kl_rewards_norm,
                        metric_label="KL Reward (normalized)",
                        file_prefix="kl_reward",
                    )
            self._last_kl_plot_step = step
        except Exception as exc:  # pragma: no cover - best-effort plotting
            warnings.warn(f"Failed to save reward plot at step {step}: {exc}")
        finally:
            self._clear_reward_plot_buffer()

    def _maybe_log_sce_bin_summary(
        self,
        step: int,
        prompts: list[Any],
        kl_values: list[float],
        completion_lengths: list[float],
        repeat_ratios: list[float],
        acc_values: list[float],
    ) -> None:
        summary, valid_groups, total_groups = self._build_sce_bin_summary(
            prompts,
            kl_values,
            completion_lengths,
            repeat_ratios,
            acc_values,
        )
        if total_groups > 0:
            percent = 100.0 * valid_groups / total_groups
            self.accelerator.print(
                f"[SCE bins] Using {valid_groups}/{total_groups} prompt groups ({percent:.1f}%)."
            )
        if not summary:
            return

        columns = [
            "step",
            "bin",
            "bin_label",
            "completion_length_mean",
            "repeat_ngram3_mean",
            "repeat_ngram3_max",
            "repeat_ngram3_collapse_rate",
            "accuracy_reward_mean",
        ]
        data = [
            [
                step,
                row["bin"],
                row["bin_label"],
                row["completion_length_mean"],
                row["repeat_ngram3_mean"],
                row["repeat_ngram3_max"],
                row["repeat_ngram3_collapse_rate"],
                row["accuracy_reward_mean"],
            ]
            for row in summary
        ]
        if is_wandb_available() and wandb.run is not None:
            wandb.log({"sce_bin_summary": wandb.Table(columns=columns, data=data)})
        if plt is not None:
            self._save_sce_bin_plot(step, summary)

    def _build_sce_bin_summary(
        self,
        prompts: list[Any],
        kl_values: list[float],
        completion_lengths: list[float],
        repeat_ratios: list[float],
        acc_values: list[float],
    ) -> tuple[list[dict[str, float]], int, int]:
        group_size = self.num_generations
        if group_size <= 0:
            return [], 0, 0
        # Prefer 8-bin or 7-bin summaries when num_generations is a multiple of those sizes.
        if group_size % 8 == 0:
            bin_count = 8
        elif group_size % 7 == 0:
            bin_count = 7
        else:
            bin_count = group_size
        if group_size % bin_count != 0:
            bin_count = group_size
        bin_size = group_size // bin_count
        total_prompt_groups = len(prompts) // group_size
        if total_prompt_groups <= 0:
            return [], 0, total_prompt_groups

        valid_kl: list[list[float]] = []
        valid_len: list[list[float]] = []
        valid_rep: list[list[float]] = []
        valid_acc: list[list[float]] = []
        for group_idx in range(total_prompt_groups):
            start = group_idx * group_size
            end = start + group_size
            if (
                end > len(kl_values)
                or end > len(completion_lengths)
                or end > len(repeat_ratios)
                or end > len(acc_values)
            ):
                continue
            group_prompts = prompts[start:end]
            if not group_prompts:
                continue
            first_prompt = group_prompts[0]
            if not all(prompt == first_prompt for prompt in group_prompts):
                continue
            valid_kl.append(kl_values[start:end])
            valid_len.append(completion_lengths[start:end])
            valid_rep.append(repeat_ratios[start:end])
            valid_acc.append(acc_values[start:end])

        if not valid_kl:
            return [], 0, total_prompt_groups

        kl_tensor = torch.tensor(valid_kl, dtype=torch.float32)
        len_tensor = torch.tensor(valid_len, dtype=torch.float32)
        rep_tensor = torch.tensor(valid_rep, dtype=torch.float32)
        acc_tensor = torch.tensor(valid_acc, dtype=torch.float32)

        order = torch.argsort(kl_tensor, dim=1, stable=True)
        summary: list[dict[str, float]] = []
        for bin_idx in range(bin_count):
            start = bin_idx * bin_size
            end = start + bin_size
            idx = order[:, start:end]
            bin_lengths = torch.gather(len_tensor, 1, idx).reshape(-1)
            bin_reps = torch.gather(rep_tensor, 1, idx).reshape(-1)
            bin_accs = torch.gather(acc_tensor, 1, idx).reshape(-1)

            length_mean = float(bin_lengths.mean().item()) if bin_lengths.numel() else float("nan")
            rep_mean = float(bin_reps.mean().item()) if bin_reps.numel() else float("nan")
            rep_max = float(bin_reps.max().item()) if bin_reps.numel() else float("nan")
            collapse_rate = (
                float((bin_reps >= 0.5).float().mean().item()) if bin_reps.numel() else float("nan")
            )
            if torch.isfinite(bin_accs).any():
                acc_mean = float(torch.nanmean(bin_accs).item())
            else:
                acc_mean = float("nan")

            bin_label = f"Bin {bin_idx}"
            if bin_idx == 0:
                bin_label += " (Lowest SCE)"
            elif bin_idx == bin_count - 1:
                bin_label += " (Highest SCE)"

            summary.append(
                {
                    "bin": int(bin_idx),
                    "bin_label": bin_label,
                    "completion_length_mean": length_mean,
                    "repeat_ngram3_mean": rep_mean,
                    "repeat_ngram3_max": rep_max,
                    "repeat_ngram3_collapse_rate": collapse_rate,
                    "accuracy_reward_mean": acc_mean,
                }
            )
        return summary, len(valid_kl), total_prompt_groups

    def _save_sce_bin_plot(self, step: int, summary: list[dict[str, float]]) -> None:
        if not summary:
            return
        bins = [int(row["bin"]) for row in summary]
        length_means = [row["completion_length_mean"] for row in summary]
        rep_means = [row["repeat_ngram3_mean"] for row in summary]
        rep_maxes = [row["repeat_ngram3_max"] for row in summary]
        collapse_rates = [row["repeat_ngram3_collapse_rate"] for row in summary]
        acc_means = [row["accuracy_reward_mean"] for row in summary]

        fig, axes = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)
        axes = axes.flatten()

        def plot_series(ax, values, title, ylabel, color, value_fmt: Optional[str] = None):
            ax.plot(bins, values, marker="o", color=color)
            ax.set_xticks(bins)
            ax.set_xlabel("SCE Rank Bin (Low -> High)")
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.4, linestyle=":")
            if value_fmt is None:
                value_fmt = "{:.0f}" if ylabel == "Tokens" else "{:.3f}"
            for x_val, y_val in zip(bins, values):
                if not math.isfinite(y_val):
                    continue
                ax.annotate(
                    value_fmt.format(y_val),
                    (x_val, y_val),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=color,
                )

        plot_series(axes[0], length_means, "Completion Length (mean)", "Tokens", "#1f77b4")
        plot_series(axes[1], rep_means, "Repeat Ngram-3 (mean)", "Ratio", "#ff7f0e")
        plot_series(axes[2], rep_maxes, "Repeat Ngram-3 (max)", "Ratio", "#d62728")
        plot_series(axes[3], collapse_rates, "Repeat Ngram-3 (collapse rate)", "Rate", "#2ca02c")
        plot_series(axes[4], acc_means, "Accuracy Reward (mean)", "Value", "#9467bd")
        fig.delaxes(axes[5])

        bin_count = len(summary)
        fig.suptitle(f"SCE {bin_count}-Bin Summary (step {step})", fontsize=12)
        plot_dir = self._kl_plot_dir
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"sce_bin_summary_step_{step:06d}.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        self.accelerator.print(f"[SCE bins] Saved {bin_count}-bin summary plot to {plot_path}")

    def _save_reward_heatmap(
        self,
        step: int,
        prompts: list[str],
        reward_values: list[float],
        *,
        metric_label: str,
        file_prefix: str,
    ) -> None:
        """Render and save a reward heatmap for the provided prompts."""
        group_size = self.num_generations
        total = len(prompts)
        if group_size <= 0 or total < group_size:
            return
        if total % group_size != 0:
            warnings.warn(
                f"Cannot reshape {metric_label} into prompt groups: total={total}, num_generations={group_size}."
            )
            return

        prompt_groups = []
        for group_idx in range(total // group_size):
            start = group_idx * group_size
            end = start + group_size
            raw_values = reward_values[start:end]
            tensor_values = torch.tensor(raw_values, dtype=torch.float32)
            finite_mask = torch.isfinite(tensor_values)
            if not finite_mask.any():
                continue
            finite_values = tensor_values[finite_mask]
            mean_tensor = finite_values.mean()
            std_tensor = finite_values.std(unbiased=False)
            std_val = float(std_tensor.item())
            mean_val = float(mean_tensor.item())
            if finite_values.numel() >= 2 and std_val > 0:
                centered = finite_values - mean_tensor
                kurt_tensor = centered.pow(4).mean() / (std_tensor.pow(4) + 1e-12)
                kurt_val = float(kurt_tensor.item())
            else:
                kurt_val = float("nan")
            prompt_groups.append(
                {
                    "prompt": self._format_prompt_label(prompts[start]),
                    "values": torch.nan_to_num(tensor_values, nan=0.0).tolist(),
                    "std": std_val,
                    "min": float(finite_values.min().item()),
                    "max": float(finite_values.max().item()),
                    "mean": mean_val,
                    "kurtosis": kurt_val,
                }
            )

        if not prompt_groups:
            return

        prompt_groups.sort(key=lambda entry: entry["std"], reverse=True)
        prompt_groups = prompt_groups[: self.kl_reward_plot_max_prompts]

        heatmap_data = torch.tensor([entry["values"] for entry in prompt_groups], dtype=torch.float32).numpy()
        data_min = float(heatmap_data.min())
        data_max = float(heatmap_data.max())
        if abs(data_max - data_min) < 1e-6:
            data_max = data_min + 1e-6
        num_rows = len(prompt_groups)
        fig_height = max(6.0, num_rows * 0.65 + 3.5)
        fig_width = max(14.0, group_size * 0.7 + 9.0)

        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
        grid = fig.add_gridspec(
            2,
            3,
            width_ratios=[3.0, 2.0, 2.0],
            height_ratios=[2.2, 1.3],
            wspace=0.3,
            hspace=0.35,
        )
        ax_heat = fig.add_subplot(grid[:, 0])
        ax_std = fig.add_subplot(grid[0, 1])
        ax_violin = fig.add_subplot(grid[1, 1])
        ax_ecdf = fig.add_subplot(grid[0, 2])
        ax_kurt = fig.add_subplot(grid[1, 2])

        im = ax_heat.imshow(
            heatmap_data,
            aspect="auto",
            cmap="coolwarm",
            vmin=data_min,
            vmax=data_max,
        )
        ax_heat.set_xticks(range(group_size))
        ax_heat.set_xticklabels([f"A{i+1}" for i in range(group_size)], rotation=45, ha="right")
        ax_heat.set_yticks(range(num_rows))
        ax_heat.set_yticklabels([entry["prompt"] for entry in prompt_groups], fontsize=8)
        ax_heat.set_xlabel("Completion Index")
        ax_heat.set_title(f"{metric_label} Distribution (step {step})")
        cbar = fig.colorbar(im, ax=ax_heat, pad=0.015)
        cbar.set_label(metric_label, rotation=270, labelpad=15)

        std_values = [entry["std"] for entry in prompt_groups]
        ax_std.barh(range(num_rows), std_values, color="#1f77b4")
        ax_std.set_yticks(range(num_rows))
        ax_std.set_yticklabels([])
        ax_std.set_xlabel("Std Dev")
        ax_std.grid(axis="x", linestyle=":", alpha=0.4)
        for idx, value in enumerate(std_values):
            ax_std.text(value, idx, f"{value:.3f}", va="center", ha="left", fontsize=8)
        ax_std.set_title("Intra-prompt std")

        violin_values: list[np.ndarray] = []
        violin_positions: list[int] = []
        for idx in range(group_size):
            column = heatmap_data[:, idx]
            finite_column = column[np.isfinite(column)]
            if finite_column.size == 0:
                continue
            violin_values.append(finite_column)
            violin_positions.append(idx + 1)
        if violin_values:
            parts = ax_violin.violinplot(
                violin_values,
                positions=violin_positions,
                showmeans=True,
                showextrema=False,
                widths=0.9,
            )
            for body in parts["bodies"]:
                body.set_facecolor("#9467bd")
                body.set_edgecolor("#4b3069")
                body.set_alpha(0.5)
            if "cmeans" in parts:
                parts["cmeans"].set_color("#4b3069")
            ax_violin.set_xticks(violin_positions)
            ax_violin.set_xticklabels([f"A{i}" for i in violin_positions])
        else:
            ax_violin.text(0.5, 0.5, "No finite values", transform=ax_violin.transAxes, ha="center", va="center")
            ax_violin.set_xticks([])
        ax_violin.set_ylabel(metric_label)
        ax_violin.set_title("Per-completion violin", fontsize=10)
        ax_violin.grid(axis="y", linestyle=":", alpha=0.4)

        flattened = heatmap_data[np.isfinite(heatmap_data)]
        if flattened.size > 0:
            sorted_vals = np.sort(flattened)
            y_vals = np.linspace(1 / sorted_vals.size, 1.0, sorted_vals.size)
            ax_ecdf.plot(sorted_vals, y_vals, color="#2ca02c")
            mean_all = float(sorted_vals.mean())
            std_all = float(sorted_vals.std())
            median_all = float(np.median(sorted_vals))
            ax_ecdf.axvline(mean_all, color="#d62728", linestyle="--", linewidth=1, label="mean")
            ax_ecdf.axvline(median_all, color="#1f77b4", linestyle=":", linewidth=1, label="median")
            ax_ecdf.legend(fontsize=8, loc="lower right")
            ax_ecdf.text(
                0.05,
                0.05,
                f"std={std_all:.3f}",
                transform=ax_ecdf.transAxes,
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )
        else:
            ax_ecdf.text(0.5, 0.5, "No data", transform=ax_ecdf.transAxes, ha="center", va="center")
        ax_ecdf.set_title("ECDF (all completions)", fontsize=10)
        ax_ecdf.set_xlabel(metric_label)
        ax_ecdf.set_ylabel("CDF")
        ax_ecdf.grid(alpha=0.4, linestyle=":")

        kurt_values = [entry.get("kurtosis", float("nan")) for entry in prompt_groups]
        kurt_plot = [value if math.isfinite(value) else 0.0 for value in kurt_values]
        ax_kurt.barh(range(num_rows), kurt_plot, color="#ff7f0e")
        ax_kurt.set_yticks(range(num_rows))
        ax_kurt.set_yticklabels([])
        ax_kurt.set_xlabel("Kurtosis")
        ax_kurt.grid(axis="x", linestyle=":", alpha=0.4)
        for idx, value in enumerate(kurt_values):
            label = f"{value:.2f}" if math.isfinite(value) else "n/a"
            ax_kurt.text(kurt_plot[idx], idx, label, va="center", ha="left", fontsize=8)
        ax_kurt.set_title("Kurtosis (sharpness)", fontsize=10)

        plot_dir = self._kl_plot_dir
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"{file_prefix}_step_{step:06d}.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        self.accelerator.print(f"[KL plot] Saved {metric_label} heatmap to {plot_path}")

    @staticmethod
    def _format_prompt_label(prompt: Any, max_chars: int = 80) -> str:
        """Collapse whitespace and clip prompt text for plotting labels."""
        text = str(prompt).replace("\n", " ")
        text = " ".join(text.split())
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    def _maybe_log_wandb_artifacts(self) -> None:
        """Upload code snapshot as a W&B artifact when enabled via UPLOAD_WANDB_ARTIFACTS."""
        if self._wandb_artifact_logged:
            return

        upload_flag = str(os.environ.get("UPLOAD_WANDB_ARTIFACTS", "")).lower()
        if upload_flag not in {"1", "true", "yes", "y", "on"}:
            return
        if not self.accelerator.is_main_process:
            return
        if not (is_wandb_available() and wandb.run is not None):
            self.accelerator.print("[WANDB] Skipping artifact upload because no active wandb run was found.")
            return

        repo_root = Path(__file__).resolve().parents[2]
        run = wandb.run
        artifact_parts = []
        if run and run.name:
            artifact_parts.append(run.name)
        if run and run.id:
            artifact_parts.append(run.id)
        artifact_name = "-".join(artifact_parts) if artifact_parts else "training_code"
        code_artifact = wandb.Artifact(
            artifact_name,
            type="code",
            metadata={
                "run_id": run.id if run else None,
                "run_name": run.name if run else None,
                "run_url": run.get_url() if run else None,
            },
        )

        added_any = False
        for rel_path in ("src", "training_scripts", "recipes", "det_yaml"):
            target = repo_root / rel_path
            if target.exists():
                code_artifact.add_dir(str(target), name=rel_path)
                added_any = True

        if not added_any:
            self.accelerator.print("[WANDB] No code directories found for artifact upload (expected src/ and training_scripts/).")
            return

        aliases = ["latest"]
        if run and run.name:
            aliases.append(run.name)

        try:
            wandb.run.log_artifact(code_artifact, aliases=aliases)
            self._wandb_artifact_logged = True
            self.accelerator.print(
                f"[WANDB] Uploaded code artifact '{artifact_name}' (aliases={aliases}) with src/ and training_scripts/ for reproducibility."
            )
        except Exception as exc:  # best-effort; do not fail the run
            self.accelerator.print(f"[WANDB] Failed to upload code artifact: {exc}")

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
