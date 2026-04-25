import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler

from lightning_trainer.data import TinyImageNetDataModule
from lightning_trainer.model import ImageClassifier, ImageClassifierConfig
from lightning_trainer.train import setup_msvc


def main():
    setup_msvc()

    # Use standard data dir as cache dir is None in benchmark above
    data_module = TinyImageNetDataModule(
        data_dir="data/tiny_imagenet_local",
        cache_dir=None,
        batch_size=128,
        num_workers=4,
        image_size=128,
    )

    model = ImageClassifier(
        ImageClassifierConfig(
            num_classes=200,
            lr=1e-4,
            max_epochs=1,
            compile_model=True,
            use_fused_optimizer=True,
            pretrained=False,
        )
    )

    # Use PyTorchProfiler for deep performance analysis
    profiler = PyTorchProfiler(
        dirpath="outputs/profiler",
        filename="perf_trace",
        export_to_chrome=True,
        row_limit=50,
        sort_by_key="cuda_time_total",
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )

    trainer = pl.Trainer(
        max_epochs=1,
        # 60 steps ensures we capture compilation and steady-state execution
        max_steps=60,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        profiler=profiler,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
