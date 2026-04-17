from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    test_loss: float
    train_accuracy: float
    test_accuracy: float
    learning_rate: float


def run_single_timestep_epoch(
    model: nn.Module,
    dataloader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    progress_desc: str = "Epoch",
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        pbar = tqdm(dataloader, total=len(dataloader), desc=progress_desc)
        for data, labels in pbar:
            data = data.to(device)
            labels = labels.to(device)
            time_steps = data.shape[0]

            model.reset()
            if is_training:
                optimizer.zero_grad()

            batch_loss = 0.0
            spike_steps = []
            for t in range(time_steps):
                logits, _, spikes = model(data[t])
                loss_t = loss_fn(logits, labels) / time_steps
                if is_training:
                    loss_t.backward()
                batch_loss += loss_t.item()
                spike_steps.append(spikes.detach())

            if is_training:
                optimizer.step()

            predictions = torch.stack(spike_steps, dim=0).sum(dim=0).argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += batch_loss * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

            avg_loss = total_loss / max(total_samples, 1)
            avg_acc = total_correct / max(total_samples, 1)
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

    average_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return average_loss, accuracy


def ensure_run_dir(output_root: str | Path, run_id: str) -> Path:
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: str | Path, payload: dict[str, Any]):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def save_config(run_dir: str | Path, config: dict[str, Any]):
    save_json(Path(run_dir) / "config.json", config)


def append_metrics(run_dir: str | Path, metrics: EpochMetrics):
    metrics_path = Path(run_dir) / "metrics.csv"
    write_header = not metrics_path.exists()
    with metrics_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(metrics).keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(metrics))


def save_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    epoch: int,
    best_test_accuracy: float,
    run_id: str,
    config: dict[str, Any],
):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_test_accuracy": best_test_accuracy,
        "run_id": run_id,
        "config": config,
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    device: torch.device,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint

