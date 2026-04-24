# Lightning Trainer

极简的 PyTorch Lightning Tiny-ImageNet 训练框架，专注于核心功能和优化。

## 特性

- **极简架构**: 仅 3 个核心文件，~350 行代码
- **PyTorch Lightning**: 自动处理设备管理、混合精度、checkpoint
- **核心优化**: torch.compile、梯度检查点、融合优化器、channels_last

## 项目结构

```
lightning_trainer/
├── __init__.py   # 包初始化
├── data.py       # TinyImageNetDataModule (~95行)
├── model.py      # ImageClassifier (~105行)
└── train.py      # CLI入口 (~150行)

data/                          # 数据目录
├── tiny-imagenet-200/         # 完整数据集 (100k训练/10k验证)
└── tiny_imagenet_local/       # 测试子集

download_data.py               # 数据下载脚本
```

## 安装

```bash
uv sync
```

## 数据准备

```bash
# 下载完整 Tiny-ImageNet (约 240MB)
python download_data.py --method huggingface

# 或创建测试子集 (10类 x 100张)
python download_data.py --method subset --num-classes 10 --images-per-class 100
```

## 训练

```bash
# 基础训练
python -m lightning_trainer.train --data-dir data/tiny-imagenet-200

# 启用所有优化
python -m lightning_trainer.train \
    --data-dir data/tiny-imagenet-200 \
    --batch-size 128 \
    --max-epochs 10 \
    --compile \
    --gradient-checkpointing \
    --fused-optimizer

# 训练后测试
python -m lightning_trainer.train \
    --data-dir data/tiny-imagenet-200 \
    --compile \
    --test
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | `data/tiny-imagenet-200` | 数据集目录 |
| `--batch-size` | `128` | 批次大小 |
| `--num-workers` | `2` | 数据加载线程数 |
| `--model-name` | `convnext_tiny` | 模型: `convnext_tiny` 或 `resnet18` |
| `--lr` | `1e-4` | 学习率 |
| `--max-epochs` | `10` | 训练轮数 |
| `--compile` | 否 | 启用 torch.compile (需要 MSVC) |
| `--gradient-checkpointing` | 否 | 启用梯度检查点 (节省46%显存) |
| `--fused-optimizer` | 否 | 启用融合 AdamW |
| `--no-pretrained` | 否 | 不使用预训练权重 |
| `--test` | 否 | 训练后运行测试集 |

## 优化效果

| 优化 | 效果 |
|------|------|
| torch.compile | ~36% 加速 |
| 梯度检查点 | ~46% 显存节省 |
| 融合优化器 | ~3% 加速 |
| 混合精度 (默认) | ~30% 加速 |
| channels_last (默认) | ~10% 加速 |

## 环境变量 (Windows torch.compile)

```bash
# 自定义 MSVC 路径
set MSVC_PATH=F:\Program Files (x86)\vs 2026
set MSVC_VERSION=14.51.36014

# 自定义 Windows SDK 路径
set WINDOWS_SDK_PATH=C:\Program Files (x86)\Windows Kits\10
```

## 数据集结构

```
data/tiny-imagenet-200/
├── train/          # 必需 - 训练集
│   ├── n01443537/
│   │   └── images/*.JPEG
│   └── ...
├── val/            # 必需 - 验证集
│   └── ...
└── test/           # 可选 - 测试集
    └── ...
```

## 依赖

- Python >= 3.11
- PyTorch >= 2.0
- PyTorch Lightning >= 2.0
- torchvision
- triton-windows (Windows)
