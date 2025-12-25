#!/usr/bin/env python3
"""
Convert Dolly JSONL to CSV with prompt = context + instruction, keeping response.
"""

import argparse
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Dolly CSV with prompt/response columns")
    parser.add_argument(
        "--input-jsonl",
        default="/run/determined/NAS1/public/xuexiang/dataset/databricks-dolly-15k/databricks-dolly-15k.jsonl",
        help="Path to databricks-dolly-15k.jsonl",
    )
    parser.add_argument(
        "--output-csv",
        default="/run/determined/NAS1/public/xuexiang/dataset/processes-dolly-15k/train.csv",
        help="Where to write the CSV (prompt, response)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_json(args.input_jsonl, lines=True)
    for col in ("instruction", "context", "response"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in input JSONL")

    def build_prompt(row) -> str:
        inst = (row.get("instruction") or "").strip()
        ctx = (row.get("context") or "").strip()
        if ctx:
            return f"{inst}\n\nContext:\n{ctx}"
        return inst

    out_df = pd.DataFrame(
        {
            "prompt": df.apply(build_prompt, axis=1),
            "response": df["response"],
        }
    )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} rows to {out_path}")
    print("Sample row:", out_df.iloc[0].to_dict())


if __name__ == "__main__":
    main()
