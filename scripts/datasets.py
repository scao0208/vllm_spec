"""
Dataset loaders for benchmark scripts.

Supported datasets:
- gsm8k: Math word problems
- humaneval: Code generation tasks
- mbpp: Python programming problems

Auto-detects dataset type from path.
"""

import json
from pathlib import Path
from typing import List, Optional


def load_gsm8k(path: Path) -> List[str]:
    """Load GSM8K dataset from parquet file."""
    import pandas as pd
    parquet_path = path / "main" / "test-00000-of-00001.parquet"
    df = pd.read_parquet(parquet_path)
    return df["question"].tolist()


def load_humaneval(path: Path) -> List[str]:
    """Load HumanEval dataset from JSONL file."""
    # Find jsonl file
    if path.is_file():
        jsonl_path = path
    else:
        jsonl_files = list(path.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No JSONL file found in {path}")
        jsonl_path = jsonl_files[0]

    prompts = []
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data["prompt"])
    return prompts


def load_mbpp(path: Path) -> List[str]:
    """Load MBPP dataset from JSONL or parquet file."""
    import pandas as pd

    # Try parquet first
    parquet_files = list(path.glob("**/*.parquet"))
    if parquet_files:
        df = pd.read_parquet(parquet_files[0])
        if "text" in df.columns:
            return df["text"].tolist()
        elif "prompt" in df.columns:
            return df["prompt"].tolist()

    # Try JSONL
    jsonl_files = list(path.glob("*.jsonl"))
    if jsonl_files:
        prompts = []
        with open(jsonl_files[0]) as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data["text"])
        return prompts

    raise FileNotFoundError(f"No dataset file found in {path}")


def detect_and_load(dataset_path: str, num_prompts: Optional[int] = None) -> List[str]:
    """Auto-detect dataset type from path and load it.

    Args:
        dataset_path: Path to dataset directory
        num_prompts: Optional limit on number of prompts

    Returns:
        List of prompt strings
    """
    path = Path(dataset_path)
    path_lower = str(path).lower()

    # Detect dataset type from path
    if "gsm8k" in path_lower or "gsm-8k" in path_lower:
        prompts = load_gsm8k(path)
        dataset_name = "gsm8k"
    elif "humaneval" in path_lower or "human_eval" in path_lower or "human-eval" in path_lower:
        prompts = load_humaneval(path)
        dataset_name = "humaneval"
    elif "mbpp" in path_lower:
        prompts = load_mbpp(path)
        dataset_name = "mbpp"
    else:
        raise ValueError(
            f"Cannot auto-detect dataset type from path: {dataset_path}\n"
            "Path should contain 'gsm8k', 'humaneval', or 'mbpp'"
        )

    print(f"Loaded {len(prompts)} prompts from {dataset_name} dataset")

    if num_prompts is not None:
        prompts = prompts[:num_prompts]

    return prompts
