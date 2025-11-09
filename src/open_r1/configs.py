# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Optional

import trl


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    kl_reward_eta: Optional[float] = field(
        default=None,
        metadata={
            "help": "Step size eta used for the KL-based reward approximation. Defaults to learning_rate when unset."
        },
    )
    ref_reward_weight: float = field(
        default=0.5,
        metadata={
            "help": "Weight applied to the reference-model log-probability reward (negative perplexity). Set to 0 to disable."
        },
    )
    tail_repeat_reward_weight: float = field(
        default=0.0,
        metadata={
            "help": "Weight applied to the tail repetition penalty reward. Positive values penalize long repeated tokens."
        },
    )
    tail_repeat_min_run: int = field(
        default=4,
        metadata={"help": "Minimum tail run length before the repetition penalty activates."},
    )
    tail_repeat_penalty_scale: float = field(
        default=1.0,
        metadata={"help": "Linear scale applied to the tail repetition penalty once the min run is exceeded."},
    )
    semantic_reward_weight: float = field(
        default=0.0,
        metadata={"help": "Weight applied to the semantic regularizer. Set to 0.0 to disable."},
    )
    semantic_similarity_low: float = field(
        default=0.0,
        metadata={"help": "Lower bound for mean pairwise similarity. Prompts below this will be filtered."},
    )
    semantic_similarity_high: float = field(
        default=0.90,
        metadata={"help": "Upper bound for mean pairwise similarity. Prompts above this will be filtered."},
    )
    semantic_embedding_api_base: Optional[str] = field(
        default="https://api.siliconflow.cn/v1",
        metadata={
            "help": "Base URL of the external embedding service. Defaults to SiliconFlow when unset."
        },
    )
    semantic_embedding_api_key: Optional[str] = field(
        default="sk-pkkoqekaidbszexjuwchtvgybvllcftpdxbfdydwzezyklxi",
        metadata={
            "help": "API key used for the embedding service. Falls back to environment variable "
            "'SEMANTIC_EMBEDDING_API_KEY'."
        },
    )
    semantic_embedding_model: str = field(
        default="Qwen/Qwen3-Embedding-4B",
        metadata={"help": "Embedding model identifier requested from the external service."},
    )
    semantic_embedding_timeout: float = field(
        default=30.0,
        metadata={"help": "Timeout (seconds) for each embedding API call."},
    )
    semantic_embedding_batch_size: int = field(
        default=16,
        metadata={"help": "Maximum number of texts sent per embedding API request."},
    )
    semantic_embedding_dtype: str = field(
        default="float16",
        metadata={
            "help": "Torch dtype used to store semantic embeddings. "
            "Supported: 'float16', 'bfloat16', 'float32'. Lower dtypes reduce memory."
        },
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )
    kl_reward_plot_enabled: bool = field(
        default=True,
        metadata={
            "help": "When true, save per-step KL reward distribution plots to the run directory (output_dir)."
        },
    )
    kl_reward_plot_every_n_steps: int = field(
        default=5,
        metadata={
            "help": "Plot frequency in steps when KL reward plotting is enabled. Use 1 to export every logging step."
        },
    )
    kl_reward_plot_max_prompts: int = field(
        default=30,
        metadata={
            "help": "Maximum number of prompt groups to visualize per KL reward plot (sorted by standard deviation)."
        },
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )


@dataclass
class GRPOScriptArguments(trl.ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'ioi_code', 'code_format', 'soft_overlong_punishment'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
        max_completion_len (`int`):
            Maximum number of tokens in completion.
        soft_punish_cache (`int`):
            Minimum number of tokens in completion.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: [],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash", "cpp"],
        },
    )
    code_eval_test_batch_size: int = field(
        default=1,
        metadata={
            "help": "for each generation, evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases. Useful to avoid overloading the eval server + save time on wrong solutions"
        },
    )
    parallel_code_exec_per_proc: int = field(
        default=2,
        metadata={
            "help": "Number of parallel E2B code executions per process. Default of 2 is suitable for the Free Hobby tier of E2B with 8 GPUs used for training."
        },
    )

    dataset_prompt_column: str = field(
        default="prompt",
        metadata={"help": "Column to use as prompts for training."},
    )

    e2b_router_url: Optional[str] = field(
        default=None,
        metadata={"help": "URL for the E2B router. See scripts/e2b_router.py"},
    )

    morph_router_url: Optional[str] = field(
        default=None,
        metadata={"help": "URL for the MorphCloud router. See scripts/morph_router.py"},
    )

    code_provider: Optional[str] = field(
        default="e2b",
        metadata={
            "help": "Provider for code execution. Options: 'e2b', 'local', 'morph'.",
            "choices": ["e2b", "local", "morph"],
        },
    )

    ioi_provider: Optional[str] = field(
        default="piston",
        metadata={
            "help": "Provider for IOI code execution. Options: 'piston', 'morph'.",
            "choices": ["piston", "morph"],
        },
    )

    max_completion_len: int = field(
        default=16384,
        metadata={"help": "Maximum number of characters in completion."},
    )
    soft_punish_cache: int = field(
        default=4096,
        metadata={"help": "Minimum number of characters in completion."},
    )
