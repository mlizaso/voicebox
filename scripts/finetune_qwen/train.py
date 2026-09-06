"""Train a single verified Spanish Qwen voice with bounded MPS memory and resume."""

import argparse
import fcntl
import importlib.metadata
import json
import math
import random
import signal
import time
from pathlib import Path

import torch

from .checkpoint import (
    adapter_state,
    cpu_tree,
    durable_json,
    publish_checkpoint,
    read_checkpoint,
    restore_adapters,
)
from .corpus import sha256
from .data import base_identity, code_row, group_key, load_verified_corpus, select_reference
from .torch_model import fixed_speaker_tensor, install_adapters, load_local_mlx_base, reference_embedding, training_loss


def epoch_order(length: int, seed: int, epoch: int) -> list[int]:
    """Reconstruct each shuffled epoch without relying on process random state."""
    order = list(range(length))
    random.Random(seed + epoch).shuffle(order)
    return order


def validation_rows(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Use a fixed chapter-balanced validation set for checkpoint selection."""
    chapters = sorted({group_key(row) for row in rows})
    buckets = []
    for chapter in chapters:
        bucket = [row for row in rows if group_key(row) == chapter]
        # Preserve legacy book selections exactly; qualify new episode sources.
        chapter_seed = seed + chapter[1] if not chapter[0] else f"{seed}:{chapter[0]}:{chapter[1]}"
        random.Random(chapter_seed).shuffle(bucket)
        buckets.append(bucket)
    selected = []
    while buckets and len(selected) < count:
        for bucket in buckets:
            if bucket and len(selected) < count:
                selected.append(bucket.pop())
        buckets = [bucket for bucket in buckets if bucket]
    return selected


def initial_state(parent: Path, identity: dict, rank: int, training: list[dict]) -> tuple[dict, dict, dict]:
    """Start a new experiment from a completed parent's selected adapters only."""
    with (parent / ".training.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        if json.loads((parent / "status.json").read_text()).get("event") != "complete":
            raise ValueError("Warm-start requires a completed parent experiment")
        pointer = json.loads((parent / "best.json").read_text())
        saved = read_checkpoint(parent, kind="best")
        contract = json.loads((parent / "contract.json").read_text())
        if saved["contract"] != contract or contract["base"] != identity or contract["rank"] != rank:
            raise ValueError("Warm-start base, rank or parent contract mismatch")
        if (
            saved["step"] != saved["best_step"]
            or saved["step"] < 1
            or not math.isfinite(saved["best_loss"])
            or not saved["best_loss"] < saved["baseline"]["loss"]
        ):
            raise ValueError("Warm-start requires the parent's improved selected checkpoint")
        reference = next(
            (
                row
                for row in training
                if row["id"] == contract["reference_id"]
                and row["sha256"] == contract["reference_sha256"]
                and row["split"] == "train"
            ),
            None,
        )
        if reference is None:
            raise ValueError("The exact parent speaker reference must remain in verified training material")
        parent_identity = {
            "run": str(parent.resolve()),
            "contract_sha256": sha256(parent / "contract.json"),
            "best": pointer,
        }
        return saved, parent_identity, reference


def evaluate(model, tokenizer, corpus: Path, rows: list[dict], speaker: torch.Tensor) -> dict:
    """Measure held-out teacher-forced losses with equal weight per clip."""
    model.eval()
    totals = [0.0, 0.0, 0.0]
    with torch.no_grad():
        for row in rows:
            losses = training_loss(model, tokenizer, [code_row(corpus, row)], speaker)
            values = [value.item() for value in losses]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Nonfinite validation loss: {row['id']}")
            totals = [total + value for total, value in zip(totals, values, strict=True)]
    model.train()
    return dict(zip(("loss", "main", "residual"), (value / len(rows) for value in totals), strict=True))


def runtime_identity() -> dict:
    """Bind exact resume to the numerical runtime and controlling training code."""
    return {
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "qwen-tts", "transformers", "numpy")},
        "code": {
            name: sha256(Path(__file__).with_name(name))
            for name in ("torch_model.py", "train.py", "data.py", "checkpoint.py")
        },
    }


def train(args: argparse.Namespace) -> None:
    """Run training under an exclusive experiment lock."""
    if not torch.backends.mps.is_available():
        raise RuntimeError("This local trainer needs native Apple Metal/MPS access")
    if not 0 < args.memory_fraction <= 0.6:
        raise ValueError("Memory fraction must be in (0, 0.6] to leave memory for macOS")
    if min(args.rank, args.accumulate, args.epochs, args.validate_every, args.save_every, args.validation_count) < 1:
        raise ValueError("Rank, counts and intervals must be positive")
    if not 0 < args.lr < 0.001 or args.patience < 1:
        raise ValueError("Invalid learning rate or patience")
    if args.max_updates is not None and args.max_updates < 1:
        raise ValueError("The update limit must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This experiment already has a training process") from error
        _train_locked(args)


def _train_locked(args: argparse.Namespace) -> None:
    torch.mps.set_per_process_memory_fraction(args.memory_fraction)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.mps.manual_seed(args.seed)
    print("Auditing verified audio, cached codecs and base weights...", flush=True)
    splits, corpus_identity = load_verified_corpus(args.corpus)
    identity = base_identity(args.base)
    if corpus_identity["encoding"]["tokenizer_sha256"] != identity["speech_tokenizer/model.safetensors"]:
        raise ValueError("The base speech codec differs from the corpus encoder")
    parent = parent_identity = None
    if args.init_run is not None:
        if args.init_run.resolve() == args.output.resolve():
            raise ValueError("A continuation must use a new experiment directory")
        parent, parent_identity, reference = initial_state(args.init_run, identity, args.rank, splits["train"])
    else:
        reference = select_reference(splits["train"])
    valid = validation_rows(splits["validation"], args.validation_count, args.seed)
    contract = {
        "format": 1,
        "base": identity,
        "corpus": corpus_identity,
        "runtime": runtime_identity(),
        "rank": args.rank,
        "lr": args.lr,
        "accumulate": args.accumulate,
        "epochs": args.epochs,
        "seed": args.seed,
        "validate_every": args.validate_every,
        "patience": args.patience,
        "reference_id": reference["id"],
        "reference_sha256": reference["sha256"],
        "validation_ids": [row["id"] for row in valid],
    }
    if parent_identity is not None:
        contract["parent"] = parent_identity
    contract_path = args.output / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Experiment identity changed; use a new output directory")
        if not (args.output / "resume.json").exists() and (args.output / "best.json").exists():
            raise ValueError("Trained adapters have no resume checkpoint; preserve them for recovery")
    else:
        if any(args.output.glob("*.pt")) or (args.output / "resume.json").exists():
            raise ValueError("Existing checkpoints have no experiment contract")
        durable_json(contract_path, contract)

    stopped = False

    def request_stop(_number, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    print(
        f"Loading BF16 base; reference={reference['id']}, train={len(splits['train'])}, validation={len(valid)}",
        flush=True,
    )
    model, tokenizer = load_local_mlx_base(args.base, "mps")
    speaker = reference_embedding(model, Path(reference["audio"]))
    install_adapters(model, args.rank)
    if parent is not None:
        restore_adapters(model, parent["adapters"])
        speaker = fixed_speaker_tensor(model, parent["speaker"])
        print(
            f"Initialized from parent selected update {parent['step']}; new optimizer and validation baseline",
            flush=True,
        )
        del parent
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    position = epoch = step = stale = 0
    best_loss = None
    best_step = 0
    baseline = None

    if (args.output / "resume.json").exists():
        saved = read_checkpoint(args.output)
        if saved["contract"] != contract:
            raise ValueError("Resume checkpoint belongs to another experiment")
        restore_adapters(model, saved["adapters"])
        optimizer.load_state_dict(saved["optimizer"])
        speaker = fixed_speaker_tensor(model, saved["speaker"])
        torch.set_rng_state(saved["rng_cpu"])
        torch.mps.set_rng_state(saved["rng_mps"])
        position, epoch, step = saved["position"], saved["epoch"], saved["step"]
        best_loss, best_step, stale = saved["best_loss"], saved["best_step"], saved["stale"]
        baseline = saved["baseline"]
        print(f"Resumed optimizer at update {step}, epoch {epoch + 1}, position {position}", flush=True)
        del saved

    def snapshot(include_optimizer: bool) -> dict:
        state = {
            "contract": contract,
            "adapters": adapter_state(model),
            "speaker": cpu_tree(speaker),
            "step": step,
            "epoch": epoch,
            "position": position,
            "best_loss": best_loss,
            "best_step": best_step,
            "stale": stale,
            "baseline": baseline,
        }
        if include_optimizer:
            state.update(
                optimizer=cpu_tree(optimizer.state_dict()),
                rng_cpu=torch.get_rng_state(),
                rng_mps=torch.mps.get_rng_state(),
            )
        return state

    def record(value: dict) -> None:
        value = {"step": step, "epoch": epoch, "position": position, **value}
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
        durable_json(args.output / "status.json", value)
        print(json.dumps(value), flush=True)

    if baseline is None:
        baseline = evaluate(model, tokenizer, args.corpus, valid, speaker)
        record({"event": "baseline", "validation": baseline})
        publish_checkpoint(args.output, snapshot(True))
    print(f"Training {sum(parameter.numel() for parameter in trainable):,} adapter parameters", flush=True)
    model.train()
    started = time.monotonic()
    try:
        while epoch < args.epochs and stale < args.patience and not stopped:
            if args.max_updates is not None and step >= args.max_updates:
                break
            order = epoch_order(len(splits["train"]), args.seed, epoch)
            batch = order[position : position + args.accumulate]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            before = time.monotonic()
            for index in batch:
                row = code_row(args.corpus, splits["train"][index])
                loss, main, residual = training_loss(model, tokenizer, [row], speaker)
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite training loss: {row['id']}")
                (loss / len(batch)).backward()
                losses.append([loss.item(), main.item(), residual.item()])
                del loss, main, residual, row
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.mps.synchronize()
            step += 1
            position += len(batch)
            epoch_finished = position == len(order)
            if epoch_finished:
                epoch += 1
                position = 0
            if step % 5 == 0 or step == 1:
                record(
                    {
                        "event": "training",
                        "train_loss": sum(row[0] for row in losses) / len(losses),
                        "gradient_norm": norm.item(),
                        "seconds_per_update": time.monotonic() - before,
                        "elapsed_seconds": time.monotonic() - started,
                        "allocated_gib": torch.mps.current_allocated_memory() / 2**30,
                        "driver_gib": torch.mps.driver_allocated_memory() / 2**30,
                    }
                )
            at_limit = args.max_updates is not None and step >= args.max_updates
            if step % args.validate_every == 0 or epoch_finished or at_limit:
                metrics = evaluate(model, tokenizer, args.corpus, valid, speaker)
                if best_loss is None or metrics["loss"] < best_loss - 0.001:
                    best_loss, best_step, stale = metrics["loss"], step, 0
                    publish_checkpoint(args.output, snapshot(False), kind="best")
                else:
                    stale += 1
                record(
                    {
                        "event": "validation",
                        "validation": metrics,
                        "best_loss": best_loss,
                        "best_step": best_step,
                        "stale": stale,
                    }
                )
                publish_checkpoint(args.output, snapshot(True))
            elif step % args.save_every == 0:
                publish_checkpoint(args.output, snapshot(True))
            torch.mps.empty_cache()
        publish_checkpoint(args.output, snapshot(True))
        complete = epoch >= args.epochs or stale >= args.patience
        record(
            {
                "event": "complete" if complete else "paused",
                "best_loss": best_loss,
                "best_step": best_step,
                "baseline": baseline,
            }
        )
    except Exception as error:
        # The last committed checkpoint remains valid even if the in-memory
        # optimizer stopped mid-update. Never serialize half an update as resume.
        record({"event": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--validation-count", type=int, default=64)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--memory-fraction", type=float, default=0.45)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--init-run", type=Path, help="Completed parent run; starts a new identity-bound experiment")
    train(parser.parse_args())
