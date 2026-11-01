"""Small classifiers used to isolate optimization choices."""

from torch import nn


class LogisticRegression(nn.Module):
    """Binary linear classifier returning logits for stable BCE loss."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


class MLP(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], num_classes: int):
        super().__init__()
        input_dim = 1
        for dimension in input_shape:
            input_dim *= dimension
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def build_model(dataset_name: str, input_shape: tuple[int, ...], num_classes: int):
    if dataset_name in {"synthetic", "a9a"}:
        return LogisticRegression(input_shape[0])
    if dataset_name == "fashion_mnist":
        return MLP(input_shape, num_classes)
    if dataset_name == "cifar10":
        try:
            from torchvision.models import resnet18
        except (ImportError, RuntimeError) as error:
            raise RuntimeError("CIFAR-10 requires the torchvision vision extra") from error
        model = resnet18(weights=None, num_classes=num_classes)
        # CIFAR-10 is 32x32; the ImageNet stem would downsample it too early.
        from torch import nn
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    raise ValueError(f"Unknown dataset: {dataset_name}")
