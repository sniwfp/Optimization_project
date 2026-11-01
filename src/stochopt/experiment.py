"""Training and evaluation with an untouched final test set."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .data import DatasetBundle
from .models import build_model
from .optimizers import MomentumSGD
from .samplers import SAMPLERS


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    optimizer: str = "custom"
    sampler: str = "reshuffle"
    scheduler: str = "none"
    batch_size: int = 128
    epochs: int = 10
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cpu"
    train_limit: int | None = None


def _build_scheduler(name: str, optimizer, epochs: int):
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                         eta_min=0.0)
    if name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                         factor=0.5, patience=2)
    raise ValueError(f"Unknown scheduler: {name}")


def _epoch(model, loader, criterion, device, optimizer=None, extra_metrics=False):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    all_scores = []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                optimizer.step()
            predictions = (logits >= 0).long() if logits.ndim == 1 else logits.argmax(dim=1)
            total_loss += loss.item() * x.shape[0]
            correct += (predictions == y.long()).sum().item()
            total += x.shape[0]
            if extra_metrics:
                all_predictions.extend(predictions.cpu().tolist())
                all_targets.extend(y.long().cpu().tolist())
                if logits.ndim == 1:
                    all_scores.extend(logits.detach().cpu().tolist())
    metrics = {"loss": total_loss / total, "accuracy": correct / total}
    if extra_metrics:
        from sklearn.metrics import f1_score, roc_auc_score

        metrics["f1"] = float(f1_score(all_targets, all_predictions,
                                        average="binary" if all_scores else "macro",
                                        zero_division=0))
        if all_scores:
            metrics["roc_auc"] = (float(roc_auc_score(all_targets, all_scores))
                                  if len(set(all_targets)) > 1 else None)
    return metrics


def run_experiment(bundle: DatasetBundle, config: RunConfig) -> dict:
    """Run one seeded configuration and evaluate test at its best validation epoch."""
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if config.train_limit is not None and config.train_limit < 1:
        raise ValueError("train_limit must be positive")
    if config.optimizer not in {"custom", "torch"}:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    if config.sampler not in SAMPLERS:
        raise ValueError(f"Unknown sampler: {config.sampler}")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = build_model(config.dataset, bundle.input_shape, bundle.num_classes).to(config.device)
    criterion = nn.BCEWithLogitsLoss() if bundle.num_classes == 2 else nn.CrossEntropyLoss()
    optimizer_class = MomentumSGD if config.optimizer == "custom" else torch.optim.SGD
    optimizer = optimizer_class(model.parameters(), lr=config.lr,
                                momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = _build_scheduler(config.scheduler, optimizer, config.epochs)

    train_dataset = bundle.train
    if config.train_limit is not None:
        chosen = torch.randperm(
            len(train_dataset), generator=torch.Generator().manual_seed(config.seed)
        )[:config.train_limit].tolist()
        train_dataset = Subset(train_dataset, sorted(chosen))
    sampler_class = SAMPLERS[config.sampler]
    sampler = (sampler_class(train_dataset) if config.sampler == "sequential"
               else sampler_class(train_dataset, seed=config.seed))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler)
    validation_loader = DataLoader(bundle.validation, batch_size=config.batch_size)
    test_loader = DataLoader(bundle.test, batch_size=config.batch_size)

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    started = perf_counter()
    for epoch_number in range(1, config.epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = _epoch(model, train_loader, criterion, config.device, optimizer)
        validation_metrics = _epoch(model, validation_loader, criterion, config.device)
        if validation_metrics["loss"] < best_val_loss:
            best_val_loss = validation_metrics["loss"]
            best_epoch = epoch_number
            best_state = deepcopy(model.state_dict())
        if scheduler is not None:
            if config.scheduler == "plateau":
                scheduler.step(validation_metrics["loss"])
            else:
                scheduler.step()
        history.append({"epoch": epoch_number, "learning_rate": learning_rate,
                        "train": train_metrics, "validation": validation_metrics})

    training_seconds = perf_counter() - started
    model.load_state_dict(best_state)
    test_metrics = _epoch(model, test_loader, criterion, config.device,
                          extra_metrics=True)
    return {"config": asdict(config), "history": history,
            "best_epoch": best_epoch, "best_validation_loss": best_val_loss,
            "test": test_metrics, "training_seconds": training_seconds,
            "sizes": {"train": len(train_dataset), "validation": len(bundle.validation),
                      "test": len(bundle.test)}}
