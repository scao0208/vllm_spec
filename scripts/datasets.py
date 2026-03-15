"""
Dataset loaders for benchmark scripts.

Supported datasets:
- gsm8k: Math word problems
- humaneval: Code generation tasks
- mbpp: Python programming problems
- longbench: Long-context QA (LongBench v2, filtered < 128K tokens)

Auto-detects dataset type from path.
"""

import json
from pathlib import Path
from typing import List, Optional


def _find_parquet(path: Path, split: str = "test") -> Path:
    """Find a parquet file in the dataset directory, preferring the given split."""
    parquet_files = list(path.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet file found in {path}")
    # Prefer the requested split
    for f in parquet_files:
        if f.name.startswith(f"{split}-"):
            return f
    return parquet_files[0]


def load_gsm8k(path: Path) -> List[str]:
    """Load GSM8K dataset from parquet file."""
    import pandas as pd
    parquet_path = _find_parquet(path, "test")
    df = pd.read_parquet(parquet_path)
    return df["question"].tolist()


def load_humaneval(path: Path) -> List[str]:
    """Load HumanEval dataset from parquet or JSONL file."""
    import pandas as pd

    # Try parquet first
    parquet_files = list(path.glob("**/*.parquet"))
    if parquet_files:
        pf = _find_parquet(path, "test")
        df = pd.read_parquet(pf)
        if "prompt" in df.columns:
            return df["prompt"].tolist()

    # Try JSONL
    if path.is_file():
        jsonl_path = path
    else:
        jsonl_files = list(path.glob("**/*.jsonl"))
        if jsonl_files:
            jsonl_path = jsonl_files[0]
        else:
            raise FileNotFoundError(f"No parquet or JSONL file found in {path}")

    prompts = []
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data["prompt"])
    return prompts


def load_mbpp(path: Path) -> List[str]:
    """Load MBPP dataset from parquet or JSONL file."""
    import pandas as pd

    # Try parquet first
    parquet_files = list(path.glob("**/*.parquet"))
    if parquet_files:
        pf = _find_parquet(path, "test")
        df = pd.read_parquet(pf)
        if "text" in df.columns:
            return df["text"].tolist()
        elif "prompt" in df.columns:
            return df["prompt"].tolist()

    # Try JSONL
    jsonl_files = list(path.glob("**/*.jsonl"))
    if jsonl_files:
        prompts = []
        with open(jsonl_files[0]) as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data["text"])
        return prompts

    raise FileNotFoundError(f"No dataset file found in {path}")


def load_longbench(path: Path, max_tokens: Optional[int] = None) -> List[str]:
    """Load LongBench v2 dataset from data.json.

    Expects a JSON file with list of objects containing 'context' and 'question'.
    Supports both the filtered version (longbench_v2_filtered/data.json with
    'est_tokens' field) and the raw version (data.json without token counts).

    Args:
        path: Path to directory containing data.json, or direct path to json.
        max_tokens: Optional upper bound on estimated token count per prompt.
                    Only works with filtered data that has 'est_tokens' field.

    Returns:
        List of prompt strings (context + question).
    """
    if path.is_file() and path.suffix == ".json":
        json_path = path
    else:
        json_path = path / "data.json"
        if not json_path.exists():
            # Search for any json file
            json_files = list(path.glob("**/*.json"))
            if not json_files:
                raise FileNotFoundError(f"No JSON file found in {path}")
            json_path = json_files[0]

    with open(json_path) as f:
        data = json.load(f)

    prompts = []
    for item in data:
        # Skip if over token limit (when est_tokens is available)
        if max_tokens and "est_tokens" in item:
            if item["est_tokens"] > max_tokens:
                continue

        context = item.get("context", "")
        question = item.get("question", "")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        prompts.append(prompt)

    return prompts


def detect_and_load(dataset_path: str, num_prompts: Optional[int] = None,
                    max_tokens: Optional[int] = None) -> List[str]:
    """Auto-detect dataset type from path and load it.

    Args:
        dataset_path: Path to dataset directory
        num_prompts: Optional limit on number of prompts
        max_tokens: Optional max tokens per prompt (longbench only)

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
    elif "longbench" in path_lower:
        prompts = load_longbench(path, max_tokens=max_tokens)
        dataset_name = "longbench_v2"
    else:
        raise ValueError(
            f"Cannot auto-detect dataset type from path: {dataset_path}\n"
            "Path should contain 'gsm8k', 'humaneval', 'mbpp', or 'longbench'"
        )

    print(f"Loaded {len(prompts)} prompts from {dataset_name} dataset")

    if num_prompts is not None:
        prompts = prompts[:num_prompts]

    return prompts
