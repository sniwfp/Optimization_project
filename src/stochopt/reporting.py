"""Machine-readable experiment outputs and a compact comparison chart."""

import csv
import hashlib
import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def run_label(result: dict) -> str:
    config = result["config"]
    return (f"{config['optimizer']}/{config['sampler']}/"
            f"{config['scheduler']}/batch={config['batch_size']}")


def save_results(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "run", "best_epoch", "validation_loss", "test_loss",
            "test_accuracy", "test_f1", "test_roc_auc", "training_seconds",
            "train_size", "validation_size", "test_size"
        ])
        writer.writeheader()
        for result in results:
            writer.writerow({
                "run": run_label(result),
                "best_epoch": result["best_epoch"],
                "validation_loss": result["best_validation_loss"],
                "test_loss": result["test"]["loss"],
                "test_accuracy": result["test"]["accuracy"],
                "test_f1": result["test"]["f1"],
                "test_roc_auc": result["test"].get("roc_auc"),
                "training_seconds": result["training_seconds"],
                "train_size": result["sizes"]["train"],
                "validation_size": result["sizes"]["validation"],
                "test_size": result["sizes"]["test"],
            })


def save_environment(output_dir: Path, dataset_name: str, data_dir: Path,
                     command: list[str]) -> None:
    """Record versions and the local a9a checksum for later reproduction."""
    packages = {}
    for package in ("torch", "torchvision", "numpy", "scikit-learn", "matplotlib"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    details = {"python": sys.version.split()[0], "platform": platform.platform(),
               "packages": packages, "command": command,
               "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}
    if dataset_name == "a9a":
        digest = hashlib.sha256()
        with (data_dir / "a9a").open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        details["a9a_sha256"] = digest.hexdigest()
    (output_dir / "environment.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def save_plot(results: list[dict], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for result in results:
        history = result["history"]
        epochs = [row["epoch"] for row in history]
        label = run_label(result)
        axes[0].plot(epochs, [row["validation"]["loss"] for row in history], label=label)
        axes[1].plot(epochs, [row["validation"]["accuracy"] for row in history], label=label)
        axes[2].plot(epochs, [row["learning_rate"] for row in history], label=label)
    for axis, title, ylabel in zip(axes,
                                  ["Validation loss", "Validation accuracy", "Learning rate"],
                                  ["Loss", "Accuracy", "Learning rate"]):
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=160)
    plt.close(fig)
