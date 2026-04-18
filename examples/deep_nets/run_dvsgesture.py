from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import tonic
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = THIS_DIR.parent
REPO_ROOT = EXAMPLES_DIR.parent
for path in (REPO_ROOT, EXAMPLES_DIR, THIS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from load_dvsgesture import get_dvsgesture_dataset
from model import build_vgg_spiking_gamma_net
from training import (
    EpochMetrics,
    append_metrics,
    ensure_run_dir,
    load_checkpoint,
    run_single_timestep_epoch,
    save_checkpoint,
    save_config,
    save_json,
)
from spikinggamma.spike.base import SpikeConfig, compute_linspace_bucket_decay_rates
from spikinggamma.spike.sigma_delta import SigmaDeltaSpikeFunction


def parse_args():
    parser = argparse.ArgumentParser(description="Train a VGG-style SpikingGamma model on DVS Gesture.")
    parser.add_argument("--output-root", type=Path, default=THIS_DIR / "runs" / "dvsgesture")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Path to a saved checkpoint such as last.pt")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--architecture", type=str, default="vgg6")
    parser.add_argument("--classifier-hidden-dims", type=int, nargs="*", default=[])
    parser.add_argument("--classifier-dropout", type=float, default=0.0)
    parser.add_argument("--classifier-norm", choices=["layernorm", "ws", "none"], default="layernorm")
    parser.add_argument("--feature-norm", choices=["layernorm", "ws", "none"], default="ws")

    parser.add_argument("--classifier-n-buckets", type=int, default=10)
    parser.add_argument("--classifier-decay-factor", type=float, default=0.1)
    parser.add_argument("--feature-bucket-decay", type=float, default=0.1)
    parser.add_argument("--spike-threshold", type=float, default=0.3)
    parser.add_argument("--spike-grad-multiplier", type=float, default=None)
    parser.add_argument("--allow-multiple-spikes", action="store_true")
    parser.add_argument("--disable-adaptive-threshold", action="store_true")

    parser.add_argument("--timestep-size", type=int, default=12000)
    parser.add_argument("--sensor-width", type=int, default=32)
    parser.add_argument("--no-cache", action="store_true")

    parser.add_argument("--scheduler", choices=["none", "step", "cosine"], default="none")
    parser.add_argument("--scheduler-step-size", type=int, default=20)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--scheduler-t-max", type=int, default=None)
    parser.add_argument("--scheduler-eta-min", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_config(args, run_id: str) -> dict:
    classifier_bucket_decay_rates = compute_linspace_bucket_decay_rates(
        args.classifier_n_buckets,
        args.classifier_decay_factor,
    )
    feature_bucket_decay_rates = [args.feature_bucket_decay]
    return {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(),
        "output_root": str(args.output_root),
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "train_batch_size": args.train_batch_size,
        "test_batch_size": args.test_batch_size,
        "num_workers": args.num_workers,
        "architecture": args.architecture,
        "classifier_hidden_dims": list(args.classifier_hidden_dims),
        "classifier_dropout": args.classifier_dropout,
        "classifier_norm": args.classifier_norm,
        "feature_norm": args.feature_norm,
        "classifier_n_buckets": args.classifier_n_buckets,
        "classifier_decay_factor": args.classifier_decay_factor,
        "feature_bucket_decay": args.feature_bucket_decay,
        "classifier_bucket_decay_rates": classifier_bucket_decay_rates.tolist(),
        "feature_bucket_decay_rates": feature_bucket_decay_rates,
        "spike_threshold": args.spike_threshold,
        "spike_grad_multiplier": args.spike_grad_multiplier,
        "allow_multiple_spikes": args.allow_multiple_spikes,
        "use_adaptive_threshold": not args.disable_adaptive_threshold,
        "timestep_size": args.timestep_size,
        "sensor_width": args.sensor_width,
        "cache": not args.no_cache,
        "scheduler": args.scheduler,
        "scheduler_step_size": args.scheduler_step_size,
        "scheduler_gamma": args.scheduler_gamma,
        "scheduler_t_max": args.scheduler_t_max,
        "scheduler_eta_min": args.scheduler_eta_min,
        "device": args.device,
    }


def main():
    args = parse_args()
    SigmaDeltaSpikeFunction.graph_mode = "none"

    device = torch.device(args.device)
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        run_id = checkpoint["run_id"]
    else:
        run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = ensure_run_dir(args.output_root, run_id)
    config = build_config(args, run_id)
    save_config(run_dir, config)

    train_dataset, test_dataset, input_shape, n_outputs, _ = get_dvsgesture_dataset(
        timestep_size=args.timestep_size,
        sensor_width=args.sensor_width,
        cache=not args.no_cache,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
    )

    classifier_bucket_decay_rates = compute_linspace_bucket_decay_rates(
        args.classifier_n_buckets,
        args.classifier_decay_factor,
    )
    feature_bucket_decay_rates = torch.tensor(
        [args.feature_bucket_decay],
        dtype=classifier_bucket_decay_rates.dtype,
    )
    spike_config = SpikeConfig(
        spike_threshold=args.spike_threshold,
        allow_multiple_spikes=args.allow_multiple_spikes,
        use_adaptive_threshold=not args.disable_adaptive_threshold,
        spike_grad_multiplier=args.spike_grad_multiplier,
    )

    model = build_vgg_spiking_gamma_net(
        input_shape=input_shape,
        n_outputs=n_outputs,
        classifier_bucket_decay_rates=classifier_bucket_decay_rates,
        feature_bucket_decay_rates=feature_bucket_decay_rates,
        spike_config=spike_config,
        architecture=args.architecture,
        classifier_hidden_dims=tuple(args.classifier_hidden_dims),
        classifier_dropout=args.classifier_dropout,
        classifier_norm=args.classifier_norm,
        feature_norm=args.feature_norm,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma,
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.scheduler_t_max or args.epochs,
            eta_min=args.scheduler_eta_min if args.scheduler_eta_min is not None else args.lr * 0.25,
        )

    start_epoch = 1
    best_test_accuracy = float("-inf")
    if args.resume is not None:
        restored = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_epoch = restored["epoch"] + 1
        best_test_accuracy = restored["best_test_accuracy"]

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_accuracy = run_single_timestep_epoch(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            progress_desc=f"Train {epoch}/{args.epochs}",
        )

        test_loss, test_accuracy = run_single_timestep_epoch(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            optimizer=None,
            progress_desc=f"Test {epoch}/{args.epochs}",
        )

        current_lr = optimizer.param_groups[0]["lr"]
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            test_loss=test_loss,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            learning_rate=current_lr,
        )
        append_metrics(run_dir, metrics)

        last_checkpoint_path = run_dir / "last.pt"
        save_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_test_accuracy=max(best_test_accuracy, test_accuracy),
            run_id=run_id,
            config=config,
        )
        torch.save(model.state_dict(), run_dir / "last_model.pt")

        if test_accuracy >= best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_checkpoint_path = run_dir / "best.pt"
            save_checkpoint(
                best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_test_accuracy=best_test_accuracy,
                run_id=run_id,
                config=config,
            )
            torch.save(model.state_dict(), run_dir / "best_model.pt")

        save_json(
            run_dir / "status.json",
            {
                "run_id": run_id,
                "epoch": epoch,
                "epochs": args.epochs,
                "best_test_accuracy": best_test_accuracy,
                "last_train_loss": train_loss,
                "last_test_loss": test_loss,
                "last_train_accuracy": train_accuracy,
                "last_test_accuracy": test_accuracy,
                "last_checkpoint": str(last_checkpoint_path),
                "best_checkpoint": str(run_dir / "best.pt"),
            },
        )

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
            f"train_acc={train_accuracy:.4f} test_acc={test_accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
