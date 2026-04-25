"""极简训练模块 - PyTorch Lightning Module."""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class ImageClassifier(LightningModule):
    """图像分类器 - 支持核心优化"""

    def __init__(
        self,
        num_classes: int = 200,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 10,
        compile_model: bool = True,
        use_gradient_checkpointing: bool = False,
        use_fused_optimizer: bool = True,
        use_channels_last: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 构建模型
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.model = convnext_tiny(weights=weights)
        self.model.classifier[2] = nn.Linear(
            self.model.classifier[2].in_features, num_classes
        )

        # 梯度检查点
        if use_gradient_checkpointing:
            enable_checkpointing = getattr(
                self.model, "gradient_checkpointing_enable", None
            )
            if enable_checkpointing is None:
                warnings.warn(
                    "Gradient checkpointing is not supported by the selected "
                    "torchvision model; continuing without it.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                enable_checkpointing()

        # torch.compile (在 setup 后应用)
        self._should_compile = compile_model
        self._compiled = False

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.use_fused_optimizer = use_fused_optimizer

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def setup(self, stage=None) -> None:
        """在设备设置后应用优化"""
        # channels_last 内存格式
        if self.hparams.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        # torch.compile
        if self._should_compile and not self._compiled:
            self.model = torch.compile(
                self.model,
                backend="inductor",
                fullgraph=True,
                mode="max-autotune",
            )
            self._compiled = True

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        if self.hparams.use_channels_last:
            images = images.to(memory_format=torch.channels_last)
        images = images.to(self.dtype)
        images.div_(255.0)
        images.sub_(self.mean).div_(self.std)
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        if self.hparams.use_channels_last:
            images = images.to(memory_format=torch.channels_last)
        images = images.to(self.dtype)
        images.div_(255.0)
        images.sub_(self.mean).div_(self.std)
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        if self.hparams.use_channels_last:
            images = images.to(memory_format=torch.channels_last)
        images = images.to(self.dtype)
        images.div_(255.0)
        images.sub_(self.mean).div_(self.std)
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        if self.use_fused_optimizer and torch.cuda.is_available():
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                fused=True,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
