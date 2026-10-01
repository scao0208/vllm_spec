"""Real training script — finetunes Llama-3.1-8B with pure PyTorch DDP.

No HF Trainer/accelerate dependency (avoids datasets module shadowing).

Usage:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py
"""
import argparse
import json
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer


class QADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, max_samples=200):
        with open(data_path) as f:
            raw = json.load(f)
        self.examples = []
        for item in raw[:max_samples]:
            ctx = item.get("context", "")[:2000]
            q = item.get("question", item.get("input", ""))
            a = item.get("answer", item.get("output", ""))
            if not a:
                a = "No answer provided."
            text = f"Context: {ctx}\n\nQuestion: {q}\n\nAnswer: {a}"
            enc = tokenizer(text, truncation=True, max_length=max_length,
                            padding="max_length", return_tensors="pt")
            ids = enc["input_ids"].squeeze(0)
            mask = enc["attention_mask"].squeeze(0)
            self.examples.append((ids, mask))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/dataset_model/model/Llama-3.1-8B-Instruct")
    parser.add_argument("--data", default="/home/dataset_model/dataset/longbench_v2_filtered/data.json")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        print(f"GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
        print(f"Model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if local_rank == 0:
        print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    model.gradient_checkpointing_enable()
    model = DDP(model, device_ids=[local_rank])

    if local_rank == 0:
        print("Loading dataset...")
    dataset = QADataset(args.data, tokenizer, max_length=args.max_length)
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if local_rank == 0:
        print(f"Dataset: {len(dataset)} examples, batch_size={args.batch_size}")
        print("Training... (Ctrl+C to stop)")

    max_epochs = args.epochs if args.epochs is not None else 10**9
    for epoch in range(max_epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        for step, (input_ids, attention_mask) in enumerate(loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=input_ids)
                loss = out.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if local_rank == 0 and step % 5 == 0:
                print(f"  epoch {epoch} step {step} loss={loss.item():.4f}", flush=True)

        if local_rank == 0:
            avg = total_loss / max(len(loader), 1)
            print(f"Epoch {epoch} avg_loss={avg:.4f}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
