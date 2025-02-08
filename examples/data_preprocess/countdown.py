"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""

import os
import argparse
from argparse import Namespace
from typing import List, TypedDict

from datasets import load_dataset
from verl.utils.hdfs_io import copy, makedirs


class CLIArgs(Namespace):
    local_dir: str = "data/countdown"
    hdfs_dir: str = None
    template_type: str = "base"


class Example(TypedDict):
    starting: List[int]
    target: int
    closest: int
    expression: str
    delta: int
    score: int
    size: int


def make_prefix(dp: Example, template_type: str = "base"):
    target = dp["target"]
    numbers = dp["starting"]
    # NOTE: also need to change reward_score/countdown.py
    if template_type == "base":
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == "qwen-instruct":
        NotImplementedError("This template is not implemented yet")
    return prefix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/countdown")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--template_type", type=str, default="base")

    args = parser.parse_args(namespace=CLIArgs())

    data_source = "countdown"
    raw_dataset = load_dataset("alexjackson17/countdown-numbers-3-8")

    train_dataset = raw_dataset["train"]
    test_dataset = raw_dataset["test"]

    def make_map_fn(split: str):
        def process_fn(example: Example, idx: int):
            question = make_prefix(example, template_type=args.template_type)
            solution = {"target": example["target"], "numbers": example["starting"]}
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
