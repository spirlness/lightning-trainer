# Tiny-ImageNet Trainer

这是一个进一步收口后的 Tiny-ImageNet 训练示例，只保留最核心的训练功能：

- 本地 `ImageFolder` 数据读取
- `ResNet18` 分类模型
- 训练 / 验证循环
- `best.pt` / `last.pt` 检查点
- 配置、日志、历史指标按运行目录保存

## 项目结构

```text
tiny_imagenet_trainer/
├── cli.py             # 命令行入口
├── config.py          # 训练配置与运行目录
├── data.py            # 本地数据集与 DataLoader
├── environment.py     # 日志、随机种子、设备选择
├── modeling.py        # ResNet18 构建
├── checkpointing.py   # best / last 检查点保存
└── trainer.py         # 训练与验证循环
tests/
main.py
```

## 数据目录

训练前先准备本地数据目录：

```text
data/tiny_imagenet_local/
├── train/
│   ├── class_001/
│   └── ...
└── val/
    ├── class_001/
    └── ...
```

程序只读取本地数据，不再负责下载、缓存导出或伪造数据。

## 运行训练

```bash
python main.py --data-dir data/tiny_imagenet_local --num-epochs 10
```

常用参数：

```bash
python main.py ^
  --data-dir data/tiny_imagenet_local ^
  --batch-size 128 ^
  --num-workers 2 ^
  --num-epochs 10 ^
  --device cuda
```

## 适合新人先看的文件

- `tiny_imagenet_trainer/config.py`
- `tiny_imagenet_trainer/data.py`
- `tiny_imagenet_trainer/trainer.py`
