#!/usr/bin/env python
"""Merge a single LoRA checkpoint into a base model."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        required=True,
        help="Base model path or HF repo id (e.g. /path/to/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to a LoRA checkpoint directory (e.g. .../ckpt/checkpoint-100)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for the merged model (default: ../merged/<ckpt-name>)",
    )
    parser.add_argument(
        "--device-map",
        default="cpu",
        help="device_map passed to from_pretrained (default: cpu)",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="torch_dtype passed to from_pretrained (default: auto)",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_false",
        dest="trust_remote_code",
        help="Disable trust_remote_code for model/tokenizer loading",
    )
    parser.set_defaults(trust_remote_code=True)
    return parser.parse_args()


def default_out_dir(ckpt_dir: Path) -> Path:
    """Place merged output next to a sibling 'merged' directory."""
    parent = ckpt_dir.parent
    if parent.name == "ckpt":
        return parent.parent / "merged" / ckpt_dir.name
    return parent / "merged" / ckpt_dir.name


def merge_checkpoint(
    base_model: str,
    tokenizer,
    ckpt_dir: Path,
    out_dir: Path,
    device_map: str,
    dtype: str,
    trust_remote_code: bool,
) -> None:
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(prefix=f"{out_dir.name}.tmp-", dir=str(out_dir.parent))
    )

    torch_dtype = resolve_dtype(dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        model = PeftModel.from_pretrained(
            model,
            str(ckpt_dir),
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        merged = model.merge_and_unload()

        # Ensure pad_token exists (Qwen 默认无 pad_token)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if merged.config.pad_token_id is None and tokenizer.pad_token_id is not None:
            merged.config.pad_token_id = tokenizer.pad_token_id

        # Move to CPU before saving to avoid GPU OOM and keep safe_serialization on CPU tensors
        merged = merged.to("cpu")
        tokenizer.save_pretrained(tmp_dir)
        merged.save_pretrained(tmp_dir, safe_serialization=True)

        tmp_dir.replace(out_dir)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.ckpt).expanduser()
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    out_dir = Path(args.out).expanduser() if args.out else default_out_dir(ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base, trust_remote_code=args.trust_remote_code
    )

    print(f"Merging {ckpt_dir} -> {out_dir}")
    merge_checkpoint(
        args.base,
        tokenizer,
        ckpt_dir,
        out_dir,
        args.device_map,
        args.dtype,
        args.trust_remote_code,
    )
    print(f"Done. Output saved to {out_dir}")


def resolve_dtype(dtype: str):
    """Map string to torch dtype; keep 'auto' passthrough."""
    if dtype.lower() == "auto":
        return "auto"

    alias = {
        "fp16": "float16",
        "f16": "float16",
        "bf16": "bfloat16",
    }
    dtype_name = alias.get(dtype.lower(), dtype)
    if hasattr(torch, dtype_name):
        return getattr(torch, dtype_name)
    raise ValueError(f"Unsupported dtype: {dtype}")


if __name__ == "__main__":
    main()
