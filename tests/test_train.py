import torch
from tiny_imagenet_trainer.train import MetricTracker, set_seed


def test_metric_tracker():
    tracker = MetricTracker()

    # Simulate a batch of 10 samples
    loss = 0.5
    preds = torch.tensor([[0.8, 0.2], [0.1, 0.9]] * 5)  # 10 predictions
    targets = torch.tensor([0, 1] * 5)  # 10 targets

    tracker.update(loss, preds, targets)

    assert tracker.total == 10
    assert tracker.correct == 10
    assert tracker.accuracy == 100.0
    assert tracker.avg_loss == 0.5

    tracker.reset()
    assert tracker.total == 0
    assert tracker.avg_loss == 0.0


def test_set_seed():
    set_seed(42)
    a = torch.rand(1)
    set_seed(42)
    b = torch.rand(1)
    assert torch.equal(a, b)
