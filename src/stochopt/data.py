"""Dataset loading and fixed train/validation/test splits."""

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import torch
from torch.utils.data import Dataset, Subset, TensorDataset


A9A_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a"


@dataclass(frozen=True)
class DatasetBundle:
    train: Dataset
    validation: Dataset
    test: Dataset
    input_shape: tuple[int, ...]
    num_classes: int


def _split(dataset: Dataset, seed: int, train_fraction: float = 0.7,
           validation_fraction: float = 0.15) -> tuple[Subset, Subset, Subset]:
    n = len(dataset)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    train_end = int(n * train_fraction)
    validation_end = train_end + int(n * validation_fraction)
    if not (0 < train_end < validation_end < n):
        raise ValueError("dataset is too small for a three-way split")
    return (Subset(dataset, sorted(indices[:train_end])),
            Subset(dataset, indices[train_end:validation_end]),
            Subset(dataset, indices[validation_end:]))


def _split_official_train(dataset: Dataset, seed: int,
                          validation_fraction: float = 0.15) -> tuple[Subset, Subset]:
    n = len(dataset)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    validation_size = int(n * validation_fraction)
    if not 0 < validation_size < n:
        raise ValueError("training dataset is too small for a validation split")
    return (Subset(dataset, sorted(indices[validation_size:])),
            Subset(dataset, indices[:validation_size]))


def load_dataset(name: str, data_dir: Path, seed: int = 42,
                 download: bool = False) -> DatasetBundle:
    """Load an offline synthetic set, LIBSVM a9a, or an official vision set.

    Downloading is opt-in. Vision datasets need the optional ``vision`` extra.
    Validation is always split from the training data; test is held out.
    """
    if name == "synthetic":
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(2400, 20, generator=generator)
        weights = torch.randn(20, generator=generator)
        logits = x @ weights + 0.8 * torch.randn(2400, generator=generator)
        y = (logits > logits.median()).float()
        dataset = TensorDataset(x, y)
        train, validation, test = _split(dataset, seed)
        return DatasetBundle(train, validation, test, (20,), 2)

    if name == "a9a":
        from sklearn.datasets import load_svmlight_file

        data_dir.mkdir(parents=True, exist_ok=True)
        file_path = data_dir / "a9a"
        if not file_path.exists():
            if not download:
                raise FileNotFoundError(
                    f"{file_path} is missing. Rerun with --download to fetch a9a."
                )
            temporary_path = file_path.with_suffix(".download")
            try:
                urlretrieve(A9A_URL, temporary_path)
                temporary_path.replace(file_path)
            finally:
                temporary_path.unlink(missing_ok=True)
        sparse_x, labels = load_svmlight_file(file_path)
        x = torch.from_numpy(sparse_x.toarray()).float()
        x = x / torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(1e-12)
        y = torch.from_numpy((labels == 1).astype("float32"))
        train, validation, test = _split(TensorDataset(x, y), seed)
        return DatasetBundle(train, validation, test, (x.shape[1],), 2)

    if name in {"fashion_mnist", "cifar10"}:
        try:
            from torchvision import datasets, transforms
        except (ImportError, RuntimeError) as error:
            raise RuntimeError(
                "Vision datasets require a matching torch/torchvision installation. "
                "Install the project with the [vision] extra."
            ) from error
        data_dir.mkdir(parents=True, exist_ok=True)
        transform = transforms.ToTensor()
        if name == "fashion_mnist":
            dataset_class = datasets.FashionMNIST
            shape = (1, 28, 28)
        else:
            dataset_class = datasets.CIFAR10
            shape = (3, 32, 32)
        official_train = dataset_class(data_dir, train=True, transform=transform,
                                       download=download)
        test = dataset_class(data_dir, train=False, transform=transform,
                             download=download)
        train, validation = _split_official_train(official_train, seed)
        return DatasetBundle(train, validation, test, shape, 10)

    raise ValueError(f"Unknown dataset: {name}")
