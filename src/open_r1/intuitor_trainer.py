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
import requests
import torch
import torch.nn.functional as F
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_peft_available
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_liger_kernel_available, is_rich_available, is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import (
    disable_dropout_in_model,
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

if is_wandb_available():
    import wandb

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
        self.ref_reward_weight = float(getattr(args, "ref_reward_weight", 0.0))
        self._ref_reward_enabled = self.ref_reward_weight != 0.0
        self._needs_ref_model = (self.beta != 0.0) or self._ref_reward_enabled
        self.tail_repeat_reward_weight = float(getattr(args, "tail_repeat_reward_weight", 0.0))
        self.tail_repeat_min_run = max(1, int(getattr(args, "tail_repeat_min_run", 4)))
        self.tail_repeat_penalty_scale = float(getattr(args, "tail_repeat_penalty_scale", 1.0))
        self._tail_repeat_enabled = self.tail_repeat_reward_weight != 0.0
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
        self.semantic_reward_weight = float(getattr(args, "semantic_reward_weight", 0.0))
        self.semantic_similarity_low = float(getattr(args, "semantic_similarity_low", 0.2))
        self.semantic_similarity_high = float(getattr(args, "semantic_similarity_high", 0.9))
        self.semantic_embedding_api_base = getattr(args, "semantic_embedding_api_base", None)
        self.semantic_embedding_api_key = getattr(args, "semantic_embedding_api_key", None)
        self.semantic_embedding_model = getattr(args, "semantic_embedding_model", "Qwen/Qwen3-Embedding-4B")
        self.semantic_embedding_timeout = float(getattr(args, "semantic_embedding_timeout", 30.0))
        self.semantic_embedding_batch_size = max(1, int(getattr(args, "semantic_embedding_batch_size", 16)))
        embedding_dtype_str = getattr(args, "semantic_embedding_dtype", "float16")
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if isinstance(embedding_dtype_str, torch.dtype):
            self.semantic_embedding_dtype = embedding_dtype_str
        else:
            lowered = str(embedding_dtype_str).lower()
            if lowered not in dtype_map:
                raise ValueError(
                    f"Unsupported semantic_embedding_dtype '{embedding_dtype_str}'. "
                    "Use one of: float16, bfloat16, float32."
                )
            self.semantic_embedding_dtype = dtype_map[lowered]
        if self.semantic_similarity_low > self.semantic_similarity_high:
            self.semantic_similarity_low, self.semantic_similarity_high = (
                self.semantic_similarity_high,
                self.semantic_similarity_low,
            )
        self.semantic_similarity_filter_enabled = (
            self.semantic_similarity_low > 0.0 or self.semantic_similarity_high < 1.0
        )
        kl_eta = getattr(args, "kl_reward_eta", None)
        if kl_eta is None:
            kl_eta = getattr(args, "learning_rate", None)
        if kl_eta is None:
            kl_eta = 1.0
        self.kl_reward_eta = float(kl_eta)
        self._kl_reward_params_cache = None
        self.kl_reward_diversity_weight = float(getattr(args, "kl_reward_diversity_weight", 0.1))
        self.kl_reward_diversity_temperature = float(getattr(args, "kl_reward_diversity_temperature", 0.1))
        self.kl_reward_diversity_epsilon = float(getattr(args, "kl_reward_diversity_epsilon", 1e-8))
        # import pdb; pdb.set_trace()
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
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
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
            "ref_reward": deque(maxlen=maxlen),
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
        last_hidden_state = unwrapped_model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    def _compute_completion_embeddings_remote(
        self,
        completions_text: list[str],
        completion_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        api_key = self.semantic_embedding_api_key or os.environ.get("SEMANTIC_EMBEDDING_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Semantic embedding API key is required. Provide `semantic_embedding_api_key` in the config "
                "or set the `SEMANTIC_EMBEDDING_API_KEY` environment variable."
            )

        base_url = (self.semantic_embedding_api_base or "https://api.siliconflow.cn/v1").rstrip("/")
        endpoint = f"{base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        valid_mask = (completion_mask.sum(dim=1) > 0).cpu()
        device = completion_mask.device

        embeddings_list: list[torch.Tensor] = []
        indices: list[int] = []

        batch_size = max(1, self.semantic_embedding_batch_size)
        buffer_inputs: list[str] = []
        buffer_indices: list[int] = []

        def flush_buffer() -> None:
            if not buffer_inputs:
                return
            vectors = self._request_remote_embeddings(endpoint, headers, buffer_inputs)
            for idx, vec in zip(buffer_indices, vectors):
                vec_tensor = torch.tensor(
                    vec,
                    dtype=self.semantic_embedding_dtype,
                    device="cpu",
                )
                embeddings_list.append(vec_tensor)
                indices.append(idx)
            buffer_inputs.clear()
            buffer_indices.clear()

        for idx, text in enumerate(completions_text):
            if not valid_mask[idx]:
                continue
            clean_text = text if text.strip() else " "
            buffer_inputs.append(clean_text)
            buffer_indices.append(idx)
            if len(buffer_inputs) >= batch_size:
                flush_buffer()
        flush_buffer()

        if embeddings_list:
            embed_dim = embeddings_list[0].size(0)
        else:
            embed_dim = 1

        embeddings = torch.zeros(
            len(completions_text),
            embed_dim,
            dtype=self.semantic_embedding_dtype,
        )
        for idx, vec in zip(indices, embeddings_list):
            embeddings[idx, : vec.size(0)] = vec

        return embeddings, valid_mask

    def _request_remote_embeddings(
        self,
        endpoint: str,
        headers: dict[str, str],
        inputs: list[str],
    ) -> list[list[float]]:
        payload = {
            "model": self.semantic_embedding_model,
            "input": inputs,
        }
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.semantic_embedding_timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Embedding request failed: {exc}") from exc

        if response.status_code >= 400:
            raise RuntimeError(
                f"Embedding request failed with status {response.status_code}: {response.text}"
            )

        data = response.json()
        if "data" not in data:
            raise RuntimeError("Embedding response missing 'data' field.")

        entries = sorted(data["data"], key=lambda item: item.get("index", 0))
        return [entry["embedding"] for entry in entries]

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
                input_ids=input_ids_batch, attention_mask=attention_mask_batch, logits_to_keep=logits_to_keep + 1
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
    ) -> torch.Tensor:
        trainable_params = self._get_kl_reward_parameters(model)
        if len(trainable_params) == 0:
            return torch.zeros(
                prompt_ids.size(0),
                device=prompt_ids.device,
                dtype=torch.float32,
            )

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

        rewards = []
        model_device = prompt_ids.device
        for idx in range(input_ids.size(0)):
            model.zero_grad(set_to_none=True)
            single_input_ids = input_ids[idx : idx + 1]
            single_attention_mask = attention_mask[idx : idx + 1]
            single_labels = labels[idx : idx + 1]

            if completion_mask[idx].sum() <= 0:
                rewards.append(torch.zeros(1, device=model_device, dtype=torch.float32).squeeze(0))
                continue

            with torch.enable_grad():
                # Use bf16 autocast to reduce activation memory during KL gradient computation
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=single_input_ids,
                        attention_mask=single_attention_mask,
                        labels=single_labels,
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
            approx_kl = grad_norm
            rewards.append((-approx_kl).squeeze(0))

        return torch.stack(rewards).to(device=model_device, dtype=torch.float32)

    def _compute_kl_reward_diversity_bonus(
        self, grouped_rewards: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encourages the per-prompt KL rewards to avoid a uniform distribution by adding a log-ratio bonus.

        Returns:
            bonuses (`torch.Tensor`): Shape matches grouped_rewards. Each entry is added to the corresponding reward.
            mean_kl (`torch.Tensor`): Mean KL(q || uniform) across prompts for logging.
            mean_entropy (`torch.Tensor`): Mean entropy of q across prompts for logging.
        """

        if grouped_rewards.numel() == 0:
            zero = grouped_rewards.new_zeros(1)
            return grouped_rewards.clone(), zero, zero

        temperature = max(float(self.kl_reward_diversity_temperature), 1e-6)
        epsilon = float(self.kl_reward_diversity_epsilon)
        scaled = grouped_rewards / temperature
        scaled = scaled - scaled.max(dim=1, keepdim=True).values
        probs = torch.softmax(scaled, dim=1)
        probs = probs.clamp_min(epsilon)
        log_probs = probs.log()
        num_generations = grouped_rewards.size(1)
        log_uniform = -math.log(max(1, num_generations))
        log_ratio = log_probs - log_uniform
        bonuses = self.kl_reward_diversity_weight * log_ratio
        kl_per_prompt = (probs * log_ratio).sum(dim=1)
        entropy_per_prompt = -(probs * log_probs).sum(dim=1)
        mean_kl = kl_per_prompt.mean()
        mean_entropy = entropy_per_prompt.mean()
        # import pdb; pdb.set_trace()
        return bonuses, mean_kl, mean_entropy

    def _compute_tail_repeat_reward(self, completion_ids: torch.Tensor, completion_mask: torch.Tensor) -> Optional[torch.Tensor]:
        if not self._tail_repeat_enabled:
            return None
        device = completion_ids.device
        penalties = torch.zeros(completion_ids.size(0), device=device, dtype=torch.float32)
        lengths = completion_mask.sum(dim=1).to(torch.int64)
        min_run = self.tail_repeat_min_run
        scale = self.tail_repeat_penalty_scale

        for idx in range(completion_ids.size(0)):
            seq_len = int(lengths[idx].item())
            if seq_len <= 0:
                continue
            tokens = completion_ids[idx, :seq_len]
            last_token = tokens[-1]
            run_length = 1
            for pos in range(seq_len - 2, -1, -1):
                if tokens[pos] == last_token:
                    run_length += 1
                else:
                    break
            excess = run_length - min_run + 1
            if excess > 0:
                penalties[idx] = -scale * excess
        return penalties

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
        # 计算kl奖励
        kl_rewards = self._compute_gradient_kl_reward(
            self.model, prompt_ids, prompt_mask, completion_ids, completion_mask
        ).detach()
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

        ref_logprob_rewards = None
        if self._ref_reward_enabled and ref_per_token_logps is not None:
            completion_lengths = completion_mask.sum(dim=1).clamp(min=1).float()
            ref_logprob_rewards = (
                (ref_per_token_logps * completion_mask).sum(dim=1) / completion_lengths
            ).detach()
        tail_repeat_rewards = self._compute_tail_repeat_reward(completion_ids, completion_mask)
        if tail_repeat_rewards is not None:
            tail_repeat_rewards = tail_repeat_rewards.detach()

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        semantic_filter_mask: Optional[torch.Tensor] = None
        semantic_filter_stats: Optional[dict[str, torch.Tensor]] = None
        # === 计算当前组内的相似度，判断是否需要过滤 ===
        if self.semantic_similarity_filter_enabled and len(prompts) % self.num_generations == 0:
            embeddings = None
            valid_mask = None
            try:
                embeddings, valid_mask = self._compute_completion_embeddings_remote(completions_text, completion_mask)
            except Exception as exc:
                warnings.warn(
                    f"Semantic embedding computation failed with error '{exc}'. "
                    "Semantic similarity filter will be skipped for this batch."
                )
            if embeddings is not None and valid_mask is not None:
                embeddings_cpu = embeddings.cpu()
                valid_mask_cpu = valid_mask.cpu()
                group_count = len(prompts) // self.num_generations
                embedding_dim = embeddings_cpu.size(-1)
                group_embeddings = embeddings_cpu.view(group_count, self.num_generations, embedding_dim)
                group_valid_mask = valid_mask_cpu.view(group_count, self.num_generations)
                group_mean_similarity = torch.full(
                    (group_count,), float("nan"), dtype=torch.float32
                )

                for group_idx in range(group_count):
                    valid_entries = group_valid_mask[group_idx]
                    valid_count = int(valid_entries.sum().item())
                    if valid_count < 2:
                        continue
                    valid_embeds = group_embeddings[group_idx][valid_entries].contiguous().to(dtype=torch.float32)
                    valid_embeds = F.normalize(valid_embeds, p=2, dim=-1)
                    similarity_matrix = valid_embeds @ valid_embeds.T
                    triu_idx = torch.triu_indices(valid_count, valid_count, offset=1)
                    mean_similarity = similarity_matrix[triu_idx[0], triu_idx[1]].mean()
                    group_mean_similarity[group_idx] = mean_similarity

                group_filter = torch.ones(group_count, dtype=torch.bool)
                valid_similarity = ~torch.isnan(group_mean_similarity)
                if self.semantic_similarity_low > 0.0:
                    group_filter[valid_similarity] &= (
                        group_mean_similarity[valid_similarity] >= self.semantic_similarity_low
                    )
                if self.semantic_similarity_high < 1.0:
                    group_filter[valid_similarity] &= (
                        group_mean_similarity[valid_similarity] <= self.semantic_similarity_high
                    )

                group_filter_device = group_filter.to(device=device)
                semantic_filter_mask = group_filter_device.repeat_interleave(self.num_generations)
                semantic_filter_stats = {
                    "group_mean_similarity": group_mean_similarity.to(device=device), # 平均的相似度
                    "group_filter": group_filter_device,
                }
        if semantic_filter_mask is not None:
            self.accelerator.wait_for_everyone()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
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
        # === 计算得到 semantic_mask_local，别的不管 ===
        semantic_mask_local = None
        if semantic_filter_mask is not None:
            mask = semantic_filter_mask.to(device=device)
            rewards_per_func = torch.where(mask.unsqueeze(1), rewards_per_func, torch.zeros_like(rewards_per_func))
            semantic_mask_local = mask.float()
        filtered_group_ratio = None
        mean_similarity_stats = None
        if semantic_filter_stats is not None:
            group_filter = semantic_filter_stats["group_filter"]    # bool, shape=[num_prompt]
            total_groups = group_filter.numel()                     # int
            filtered_groups = (~group_filter).sum().float()         # float，被过滤的组
            filtered_group_ratio = (filtered_groups, torch.tensor(float(total_groups), device=device))  # 被过滤的组占比
            valid_means = semantic_filter_stats["group_mean_similarity"][~torch.isnan(
                semantic_filter_stats["group_mean_similarity"]
            )]
            if valid_means.numel() > 0:
                mean_similarity_stats = valid_means

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
        gathered_kl_rewards_raw = gather(kl_rewards).reshape(-1)    # 将所有gpu的kl_reward聚集到一起，例如单卡kl_rewards.shape=[6]，那么gathered_kl_rewards.shape=[12]
        if ref_logprob_rewards is not None:
            gathered_ref_rewards = gather(ref_logprob_rewards)
            ref_rewards_for_logging = gathered_ref_rewards.detach().cpu().tolist()
        else:
            gathered_ref_rewards = None
            ref_rewards_for_logging = None
        if tail_repeat_rewards is not None:
            gathered_tail_repeat_rewards = gather(tail_repeat_rewards)
            tail_repeat_rewards_for_logging = gathered_tail_repeat_rewards.detach().cpu().tolist()
        else:
            gathered_tail_repeat_rewards = None
            tail_repeat_rewards_for_logging = None
        if semantic_mask_local is not None:
            semantic_mask_all = gather(semantic_mask_local).to(device=device)
        else:
            semantic_mask_all = None
        # === 将原始的kl_rewards的mean和std记录下来（不再使用多样性bonus） ===
        raw_kl_rewards_by_prompt = gathered_kl_rewards_raw.view(-1, self.num_generations)
        kl_diversity_stats = None
        raw_mean_kl_rewards = raw_kl_rewards_by_prompt.mean(dim=1)
        raw_std_kl_rewards = raw_kl_rewards_by_prompt.std(dim=1)
        raw_kl_reward_record = raw_mean_kl_rewards.mean().item()
        raw_kl_reward_std_record = raw_std_kl_rewards.mean().item()

        # === 分桶（分位数）KL 奖励：按每个 prompt 的梯度范数分位切分，映射到离散档位 ===
        grad_norms_by_prompt = (-raw_kl_rewards_by_prompt).clamp_min(0.0)  # [num_prompt, num_generations]
        quantile_levels = torch.tensor([0.2, 0.4, 0.6, 0.8], device=grad_norms_by_prompt.device)
        quantiles = torch.quantile(grad_norms_by_prompt, quantile_levels, dim=1, keepdim=False)  # [4, num_prompt]
        quantiles = quantiles.transpose(0, 1)  # [num_prompt, 4]
        q20, q40, q60, q80 = [quantiles[:, i].unsqueeze(1) for i in range(4)]

        kl_buckets = torch.zeros_like(grad_norms_by_prompt)
        kl_buckets = torch.where(grad_norms_by_prompt <= q20, 2.0, kl_buckets)
        kl_buckets = torch.where((grad_norms_by_prompt > q20) & (grad_norms_by_prompt <= q40), 1.0, kl_buckets)
        kl_buckets = torch.where((grad_norms_by_prompt > q40) & (grad_norms_by_prompt <= q60), 0.0, kl_buckets)
        kl_buckets = torch.where((grad_norms_by_prompt > q60) & (grad_norms_by_prompt <= q80), -1.0, kl_buckets)
        kl_buckets = torch.where(grad_norms_by_prompt > q80, -2.0, kl_buckets)

        kl_rewards_norm_flat = kl_buckets.reshape(-1)  # shape=[num_prompt*num_generations*num_device]
        gathered_kl_rewards = kl_rewards_norm_flat
        kl_rewards_norm_for_logging = kl_rewards_norm_flat.detach().cpu().tolist()  # 保存离散化后的 KL 奖励用于文本日志
        # import pdb; pdb.set_trace()
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

        # === 只保留当前 rank 对应的数据片段 ===
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        reward_advantages_all = advantages.detach().clone() # 在切片前先备份 reward 端优势，方便后续做全局日志
        advantages = advantages[process_slice]
        # === 语义相似度过滤：语义相似度在范围之外的prompt，通过mask，将其advantage设置为0，不参与损失计算 ===
        if semantic_mask_all is not None:
            semantic_mask_local_slice = semantic_mask_all[process_slice]
        else:
            semantic_mask_local_slice = None
        # === Reference rewards：按 prompt 组统计并换算为优势 ===
        ref_reward_record = None
        ref_reward_std_record = None
        if gathered_ref_rewards is not None:
            ref_rewards_grouped = gathered_ref_rewards.view(-1, self.num_generations)
            mean_ref_rewards = ref_rewards_grouped.mean(dim=1)
            std_ref_rewards = ref_rewards_grouped.std(dim=1)
            ref_reward_record = mean_ref_rewards.mean().item()
            ref_reward_std_record = std_ref_rewards.mean().item()
            mean_ref_rewards = mean_ref_rewards.repeat_interleave(self.num_generations, dim=0)
            std_ref_rewards = std_ref_rewards.repeat_interleave(self.num_generations, dim=0)
            mean_ref_rewards_all = mean_ref_rewards.detach().clone()
            std_ref_rewards_all = std_ref_rewards.detach().clone()
            mean_ref_rewards_local = mean_ref_rewards[process_slice]
            std_ref_rewards_local = std_ref_rewards[process_slice]
            ref_advantage = (ref_logprob_rewards - mean_ref_rewards_local) / (std_ref_rewards_local + 1e-4)
            ref_advantage = ref_advantage * self.ref_reward_weight
            gathered_ref_rewards_flat = gathered_ref_rewards.view(-1)
            ref_advantages_all = (gathered_ref_rewards_flat - mean_ref_rewards_all) / (std_ref_rewards_all + 1e-4)
            ref_advantages_all = ref_advantages_all * self.ref_reward_weight
        else:
            ref_advantage = torch.zeros_like(kl_rewards)
            ref_advantages_all = torch.zeros_like(reward_advantages_all)

        # === Tail-repeat rewards：检测重复片段并生成奖励分布 ===
        tail_repeat_reward_record = None
        tail_repeat_reward_std_record = None
        if gathered_tail_repeat_rewards is not None:
            tail_repeat_grouped = gathered_tail_repeat_rewards.view(-1, self.num_generations)
            mean_tail_repeat = tail_repeat_grouped.mean(dim=1)
            std_tail_repeat = tail_repeat_grouped.std(dim=1)
            tail_repeat_reward_record = mean_tail_repeat.mean().item()
            tail_repeat_reward_std_record = std_tail_repeat.mean().item()
            mean_tail_repeat = mean_tail_repeat.repeat_interleave(self.num_generations, dim=0)
            std_tail_repeat = std_tail_repeat.repeat_interleave(self.num_generations, dim=0)
            mean_tail_repeat_all = mean_tail_repeat.detach().clone()
            std_tail_repeat_all = std_tail_repeat.detach().clone()
            mean_tail_repeat_local = mean_tail_repeat[process_slice]
            std_tail_repeat_local = std_tail_repeat[process_slice]
            tail_repeat_advantage = (tail_repeat_rewards - mean_tail_repeat_local) / (std_tail_repeat_local + 1e-4)
            tail_repeat_advantage = tail_repeat_advantage * self.tail_repeat_reward_weight
            tail_repeat_advantages_all = (gathered_tail_repeat_rewards.view(-1) - mean_tail_repeat_all) / (
                std_tail_repeat_all + 1e-4
            )
            tail_repeat_advantages_all = tail_repeat_advantages_all * self.tail_repeat_reward_weight
        else:
            tail_repeat_advantage = torch.zeros_like(kl_rewards)
            tail_repeat_advantages_all = torch.zeros_like(reward_advantages_all)

        # === KL rewards：将gathered_kl_rewards按照prompt划分，计算组内mean和std，便于计算优势 ===
        gathered_kl_rewards = gathered_kl_rewards.view(-1, self.num_generations)    # (num_prompt, num_generations)
        mean_kl_rewards = gathered_kl_rewards.mean(dim=1)   # (num_prompt) 每个 prompt 小组内部做均值
        std_kl_rewards = gathered_kl_rewards.std(dim=1)

        kl_reward_record = mean_kl_rewards.mean().item()    # 计算所有 prompt 组的平均 KL 奖励 shape=[1]
        kl_reward_std_record = std_kl_rewards.mean().item()    # 计算所有 prompt 组的平均 KL 奖励标准差
        mean_kl_rewards = mean_kl_rewards.repeat_interleave(self.num_generations, dim=0) # 会将每个元素重复 num_generations 次（num_prompt*num_generations）。如果 mean_kl_rewards = [0.5, 0.3, 0.8] 且 num_generations = 4，结果将是 [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.8, 0.8, 0.8, 0.8]
        std_kl_rewards = std_kl_rewards.repeat_interleave(self.num_generations, dim=0)

        # === 记录当前 step 的 kl_reward的mean和std ===
        mean_kl_rewards_all = mean_kl_rewards.detach().clone()
        std_kl_rewards_all = std_kl_rewards.detach().clone()

        # === process_slice 获取当前 gpu 上的样本索引范围，计算当前 gpu 的 advantages ===
        kl_rewards = kl_rewards_norm_flat[process_slice]    # kl_rewards_norm_flat是归一化之后的shape=[num_prompt*num_generation]，使用[process_slice]切出当前gpu的部分，用于计算优势
        mean_kl_rewards = mean_kl_rewards[process_slice]
        std_kl_rewards = std_kl_rewards[process_slice]
        kl_advantage = (kl_rewards - mean_kl_rewards) / (std_kl_rewards + 1e-4)        # 逐个元素相减，计算 KL 奖励的优势
        if semantic_mask_local_slice is not None:
            advantages = advantages * semantic_mask_local_slice
            kl_advantage = kl_advantage * semantic_mask_local_slice
            ref_advantage = ref_advantage * semantic_mask_local_slice
            tail_repeat_advantage = tail_repeat_advantage * semantic_mask_local_slice
        total_advantage = kl_advantage + advantages + ref_advantage + tail_repeat_advantage
        if semantic_mask_local_slice is not None:
            total_advantage = total_advantage * semantic_mask_local_slice
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

        if filtered_group_ratio is not None:
            local_filtered, local_total = filtered_group_ratio
            gathered_filtered = self.accelerator.gather_for_metrics(local_filtered)
            gathered_total = self.accelerator.gather_for_metrics(local_total)
            total_groups = gathered_total.sum().item()
            filtered_groups = gathered_filtered.sum().item()
            ratio = filtered_groups / total_groups if total_groups > 0 else 0.0
            self._metrics[mode]["semantic/filtered_ratio"].append(ratio)
        if mean_similarity_stats is not None and mean_similarity_stats.numel() > 0:
            gathered_similarity = self.accelerator.gather_for_metrics(mean_similarity_stats)
            self._metrics[mode]["semantic/mean_similarity"].append(gathered_similarity.nanmean().item())
            self._metrics[mode]["semantic/min_similarity"].append(nanmin(gathered_similarity).item())
            self._metrics[mode]["semantic/max_similarity"].append(nanmax(gathered_similarity).item())

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
        self._metrics[mode]["rewards/kl_reward_raw/mean"].append(raw_kl_reward_record)
        self._metrics[mode]["rewards/kl_reward_raw/std"].append(raw_kl_reward_std_record)
        if kl_diversity_stats is not None:
            kl_divergence_record, kl_entropy_record = kl_diversity_stats
            self._metrics[mode]["rewards/kl_reward_diversity/kl"].append(kl_divergence_record)
            self._metrics[mode]["rewards/kl_reward_diversity/entropy"].append(kl_entropy_record)
        if ref_reward_record is not None:
            self._metrics[mode]["rewards/ref_logprob/mean"].append(ref_reward_record)
            self._metrics[mode]["rewards/ref_logprob/std"].append(ref_reward_std_record)
        if tail_repeat_reward_record is not None:
            self._metrics[mode]["rewards/tail_repeat/mean"].append(tail_repeat_reward_record)
            self._metrics[mode]["rewards/tail_repeat/std"].append(tail_repeat_reward_std_record)
        # 监测 kl 的 advantage的平均值，为0才正常
        self._metrics[mode]["advantages/kl_mean_advantage"].append(self.accelerator.gather_for_metrics(kl_advantage).nanmean().item())
        if gathered_ref_rewards is not None:
            self._metrics[mode]["advantages/ref_mean_advantage"].append(
                self.accelerator.gather_for_metrics(ref_advantage).nanmean().item()
            )
        if gathered_tail_repeat_rewards is not None:
            self._metrics[mode]["advantages/tail_repeat_mean_advantage"].append(
                self.accelerator.gather_for_metrics(tail_repeat_advantage).nanmean().item()
            )
        self._metrics[mode]["advantages/total_mean_advantage"].append(self.accelerator.gather_for_metrics(total_advantage).nanmean().item())
        # Log prompt and completion texts
        gathered_prompts = gather_object(prompts_text)
        gathered_completions = gather_object(completions_text)
        self._textual_logs["prompt"].extend(gathered_prompts)
        self._textual_logs["completion"].extend(gathered_completions)
        if mode == "train" and self.kl_reward_plot_enabled:
            self._kl_plot_buffer["prompt"].extend(gathered_prompts)
            self._kl_plot_buffer["kl_reward_norm"].extend(kl_rewards_norm_for_logging)
            if ref_rewards_for_logging is not None:
                self._kl_plot_buffer["ref_reward"].extend(ref_rewards_for_logging)
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        # self._textual_logs["rewards"]["kl_reward"].extend(kl_rewards_for_logging)


        # 记录 KL/Total 的优势项到 textual_logs（全局对齐，支持多卡）：
        # 使用 gather 后的全局 KL 奖励与对应均值/方差，得到每个 completion 的 KL 优势
        gathered_kl_rewards_flat = gathered_kl_rewards.reshape(-1)
        kl_advantages_all = (gathered_kl_rewards_flat - mean_kl_rewards_all) / (std_kl_rewards_all + 1e-4)
        if semantic_mask_all is not None:
            kl_advantages_all = kl_advantages_all * semantic_mask_all
            reward_advantages_all = reward_advantages_all * semantic_mask_all
        if ref_advantages_all is not None and semantic_mask_all is not None:
            ref_advantages_all = ref_advantages_all * semantic_mask_all
        if tail_repeat_advantages_all is not None and semantic_mask_all is not None:
            tail_repeat_advantages_all = tail_repeat_advantages_all * semantic_mask_all
        # total 优势 = KL 优势 +（任务奖励优势，已在切片前缓存为 reward_advantages_all）
        total_advantages_all = reward_advantages_all + kl_advantages_all + ref_advantages_all
        if tail_repeat_advantages_all is not None:
            total_advantages_all = total_advantages_all + tail_repeat_advantages_all
        if semantic_mask_all is not None:
            total_advantages_all = total_advantages_all * semantic_mask_all
        # self._textual_logs["rewards"]["kl_advantage"].extend(kl_advantages_all.detach().cpu().tolist())
        # if ref_advantages_all is not None:
        #     self._textual_logs["rewards"]["ref_logprob_advantage"].extend(
        #         ref_advantages_all.detach().cpu().tolist()
        #     )
        # if tail_repeat_advantages_all is not None:
        #     self._textual_logs["rewards"]["tail_repeat_advantage"].extend(
        #         tail_repeat_advantages_all.detach().cpu().tolist()
        #     )
        # self._textual_logs["rewards"]["total_advantage"].extend(total_advantages_all.detach().cpu().tolist())

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
        self._metrics[mode].clear()

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
        if plt is None:
            if not self._kl_plot_warning_emitted:
                warnings.warn(
                    "matplotlib is not available in this environment; disabling reward plotting."
                )
                self._kl_plot_warning_emitted = True
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
        ref_rewards = list(self._kl_plot_buffer["ref_reward"])

        try:
            if kl_rewards_norm:
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
            if ref_rewards:
                if len(prompts) != len(ref_rewards):
                    warnings.warn("Prompt/ref buffer size mismatch; skipping ref reward plot for this step.")
                else:
                    self._save_reward_heatmap(
                        step,
                        prompts,
                        ref_rewards,
                        metric_label="Ref Logprob Reward",
                        file_prefix="ref_reward",
                    )
            self._last_kl_plot_step = step
        except Exception as exc:  # pragma: no cover - best-effort plotting
            warnings.warn(f"Failed to save reward plot at step {step}: {exc}")
        finally:
            self._clear_reward_plot_buffer()

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
