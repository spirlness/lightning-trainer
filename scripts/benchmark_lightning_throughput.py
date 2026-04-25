"""Benchmark GPU training throughput for the Lightning trainer."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from lightning_trainer.data import TinyImageNetDataModule
from lightning_trainer.model import ImageClassifier
from lightning_trainer.train import setup_msvc


class ThroughputCallback(Callback):
    def __init__(
        self,
        warmup_batches: int,
        benchmark_batches: int | None,
    ) -> None:
        self.warmup_batches = warmup_batches
        self.benchmark_batches = benchmark_batches
        self.step_start = 0.0
        self.step_times: list[float] = []
        self.sample_count = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if batch_idx >= self.warmup_batches and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.step_start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if batch_idx < self.warmup_batches:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.step_times.append(time.perf_counter() - self.step_start)
        self.sample_count += int(batch[0].shape[0])

        if (
            self.benchmark_batches is not None
            and len(self.step_times) >= self.benchmark_batches
        ):
            trainer.should_stop = True

    @property
    def throughput(self) -> float:
        total_time = sum(self.step_times)
        if total_time == 0:
            return 0.0
        return self.sample_count / total_time

    @property
    def avg_step_ms(self) -> float:
        if not self.step_times:
            return 0.0
        return sum(self.step_times) / len(self.step_times) * 1000

    def percentile_step_ms(self, percentile: float) -> float:
        if not self.step_times:
            return 0.0
        sorted_times = sorted(self.step_times)
        index = round((len(sorted_times) - 1) * percentile)
        return sorted_times[index] * 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Lightning throughput")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/tiny_imagenet_local"),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--benchmark-batches", type=int, default=30)
    parser.add_argument("--full-epoch", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-fused-optimizer", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    torch.set_float32_matmul_precision("high")
    compile_model = not args.no_compile
    if compile_model:
        setup_msvc()
        cache_dir = Path("outputs") / "torchinductor_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir.resolve()))

    data_module = TinyImageNetDataModule(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    data_module.setup("fit")

    model = ImageClassifier(
        num_classes=data_module.num_classes,
        lr=args.lr,
        compile_model=compile_model,
        use_fused_optimizer=not args.no_fused_optimizer,
        use_channels_last=True,
        pretrained=not args.no_pretrained,
    )

    benchmark_batches = None if args.full_epoch else args.benchmark_batches
    limit_train_batches = None
    if benchmark_batches is not None:
        limit_train_batches = args.warmup_batches + benchmark_batches

    callback = ThroughputCallback(args.warmup_batches, benchmark_batches)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_epochs=1,
        limit_train_batches=limit_train_batches,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        benchmark=True,
        callbacks=[callback],
    )

    started_at = time.perf_counter()
    trainer.fit(model, data_module)
    elapsed = time.perf_counter() - started_at

    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    first_window = callback.step_times[:100]
    last_window = callback.step_times[-100:]
    first_window_throughput = args.batch_size * len(first_window) / sum(first_window)
    last_window_throughput = args.batch_size * len(last_window) / sum(last_window)
    result = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "model": "convnext_tiny",
        "data_dir": str(args.data_dir),
        "cache_dir": str(args.cache_dir) if args.cache_dir is not None else None,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "num_workers": args.num_workers,
        "precision": "16-mixed",
        "compile": compile_model,
        "fused_optimizer": not args.no_fused_optimizer,
        "channels_last": True,
        "full_epoch": args.full_epoch,
        "warmup_batches": args.warmup_batches,
        "benchmark_batches": len(callback.step_times),
        "throughput_samples_per_sec": round(callback.throughput, 2),
        "avg_step_ms": round(callback.avg_step_ms, 2),
        "median_step_ms": round(statistics.median(callback.step_times) * 1000, 2),
        "p90_step_ms": round(callback.percentile_step_ms(0.90), 2),
        "p99_step_ms": round(callback.percentile_step_ms(0.99), 2),
        "first_100_throughput": round(first_window_throughput, 2),
        "last_100_throughput": round(last_window_throughput, 2),
        "peak_memory_mb": round(peak_memory_mb, 1),
        "elapsed_sec": round(elapsed, 2),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
