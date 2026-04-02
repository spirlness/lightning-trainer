# Tiny-ImageNet Trainer

这是一个适合新人学习的深度学习训练项目示例，目标不是“功能堆满”，而是把常见训练脚本整理成清晰、可扩展、容易维护的工程结构。

## 这个版本解决了什么问题

- 不再把全部逻辑塞进一个 `main.py`
- 把 `datasets` 改成可选依赖，缺失时给出明确提示
- 增加 `mock` 数据模式，先验证训练闭环，再接真实数据
- 使用更稳妥的 AMP 策略：
  CUDA + bf16 时走 `autocast(bfloat16)`，非 bf16 CUDA 时自动启用 `GradScaler`
- 真正实现了 warmup + cosine scheduler
- 日志、配置、检查点、历史指标按运行目录集中保存
- 增加最小测试，保证重构不是“只改结构不保行为”

## 项目结构

```text
tiny_imagenet_trainer/
├── cli.py             # 命令行入口
├── config.py          # 配置与运行目录
├── environment.py     # 环境、日志、随机种子
├── data.py            # 数据集与 DataLoader
├── modeling.py        # 模型构建
├── checkpointing.py   # 检查点保存与恢复
└── trainer.py         # 训练与验证循环
tests/
main.py                # 兼容入口，转发到包内 CLI
```

## 快速开始

先跑一个不依赖下载的最小闭环：

```bash
python main.py --dataset-source mock --num-epochs 1 --max-train-batches 2 --max-val-batches 1 --batch-size 4 --device cpu --no-compile --no-pretrained
```

默认推荐使用本地 imagefolder 数据源。第一次运行时，程序会从 Hugging Face 缓存把数据集导出到 `data/tiny_imagenet_local/`，之后训练会直接走本地文件读取。

如果要跑 Tiny-ImageNet 数据集，先安装可选依赖：

```bash
pip install -e .[data]
```

然后执行：

```bash
python main.py --dataset-source local --num-epochs 10
```

## 给新人的建议

- 先用 `mock` 数据把训练链路跑通，再切换真实数据
- 第一次跑 `local` 会慢一些，因为要做一次本地数据导出；第二次开始会更快
- 先改 `TrainingConfig`，再动训练逻辑
- 想扩展模型时，优先在 `modeling.py` 增加新的 builder
- 想换数据源时，优先在 `data.py` 新增 loader，而不是把逻辑塞回训练器
