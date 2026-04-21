"""Tiny-ImageNet 训练包 (初学者友好版)。
该包提供从数据加载、模型训练到评估的全套工具，专为深度学习入门者设计。
"""

from tiny_imagenet_trainer.config import TrainingConfig, parse_args

# 定义包的导出接口，方便用户导入
__all__ = ["TrainingConfig", "parse_args"]

# 版本号，便于管理和追踪
__version__ = "0.1.0"
