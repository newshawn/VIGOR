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

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name open-r1/OpenR1-Math-220k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
"""

import logging
import os
import sys

import datasets
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import (DataCollatorForCompletionOnlyLM, ModelConfig, ScriptArguments,
                 SFTTrainer, TrlParser, get_peft_config, setup_chat_format)

from open_r1.configs import SFTConfig
from open_r1.utils import get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training

logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ################
    # Load datasets
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    # 统一 response_template，便于后续掩蔽 prompt
    response_template = getattr(training_args, "response_template", None) or "<|im_start|>assistant\n"
    template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    logger.info("Response template: %s | token_ids=%s", repr(response_template), template_ids)

    ##################################################
    # Reformat Dolly-style fields -> chat-formatted text
    ##################################################
    columns = set(dataset[script_args.dataset_train_split].column_names)
    if {"instruction", "response"} <= columns:
        def _format_chat(example):
            prompt = example.get("instruction") or ""
            context = example.get("context") or ""
            if context.strip():
                prompt = prompt + "\n\nContext:\n" + context
            answer = example.get("response") or ""
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return {"text": text}

        remove_cols = dataset[script_args.dataset_train_split].column_names
        dataset = dataset.map(_format_chat, remove_columns=remove_cols)
        training_args.dataset_text_field = "text"
        # 只对 assistant 段计算 loss：ChatML 模板里 assistant 以 <|im_start|>assistant 开头
        training_args.train_on_inputs = False
        # 某些版本的 SFTConfig 没有 response_template 字段，需兼容处理
        if not hasattr(training_args, "response_template"):
            setattr(training_args, "response_template", None)
        training_args.response_template = response_template
        logger.info(
            "Detected Dolly-style columns, reformatted with chat template to 'text', "
            "set dataset_text_field='text', train_on_inputs=False, "
            f"response_template='{training_args.response_template}'."
        )
    ###################
    # Load model
    ###################
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    if tokenizer.chat_template is None:
        logger.info("No chat template provided, using ChatML.")
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    ############################
    # Initialize the SFT Trainer
    ############################
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        data_collator=data_collator,
    )

    # Debug：验证掩蔽是否生效（仅主进程）
    if trainer.accelerator.is_local_main_process:
        try:
            sample = dataset[script_args.dataset_train_split][0]
            raw_text = sample.get("text", None) or sample.get("prompt", None)
            if raw_text:
                tok = tokenizer(
                    raw_text,
                    add_special_tokens=False,
                    return_attention_mask=True,
                    truncation=True,
                    max_length=training_args.max_seq_length,
                )
                batch = data_collator(
                    [{"input_ids": tok["input_ids"], "attention_mask": tok.get("attention_mask")}]
                )
                labels = batch["labels"][0].tolist()
                non_masked = sum(1 for v in labels if v != -100)
                logger.warning(
                    "Debug sample[0]",
                    len(labels),
                    non_masked,
                    labels,
                    raw_text,
                )
            else:
                logger.warning("Debug sample[0]: no text field found, skip mask check.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Debug mask check failed: %s", exc)


    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
