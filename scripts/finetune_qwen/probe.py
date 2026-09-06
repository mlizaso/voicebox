"""Measure real MPS forward/backward updates before launching a long training run."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .torch_model import (
    install_adapters,
    load_local_mlx_base,
    reference_embedding,
    training_loss,
)


def probe(corpus: Path, model_path: Path, rank: int, batch_size: int):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Native MPS access is required for this local training probe")
    torch.manual_seed(20260905)
    started = time.monotonic()
    model, tokenizer = load_local_mlx_base(model_path, "mps")
    print(f"Loaded base in {time.monotonic() - started:.1f}s", flush=True)
    rows = [json.loads(line) for line in (corpus / "train_raw.jsonl").read_text().splitlines()]
    rows = [row for row in rows if (corpus / "codes" / f"{row['id']}.npz").exists()]
    rows = sorted(rows, key=lambda row: row["duration"], reverse=True)[:batch_size]
    for row in rows:
        with np.load(corpus / "codes" / f"{row['id']}.npz", allow_pickle=False) as saved:
            row["audio_codes"] = saved["codes"].tolist()
    speaker = reference_embedding(model, Path(rows[0]["ref_audio"]))
    install_adapters(model, rank)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}", flush=True)
    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=2e-5)
    metrics = []
    for index in range(3):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss, main, residual = training_loss(model, tokenizer, rows, speaker)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.mps.synchronize()
        result = {
            "step": index + 1,
            "loss": loss.item(),
            "main": main.item(),
            "residual": residual.item(),
            "gradient_norm": norm.item(),
            "seconds": time.monotonic() - started,
            "allocated_gib": torch.mps.current_allocated_memory() / 2**30,
            "driver_gib": torch.mps.driver_allocated_memory() / 2**30,
        }
        metrics.append(result)
        print(json.dumps(result), flush=True)
    destination = corpus.parent / f"training-probe-batch{batch_size}.json"
    destination.write_text(
        json.dumps(
            {
                "rank": rank,
                "batch_size": batch_size,
                "clips": [row["id"] for row in rows],
                "clip_seconds": sum(row["duration"] for row in rows),
                "metrics": metrics,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    probe(args.corpus, args.model, args.rank, args.batch_size)
