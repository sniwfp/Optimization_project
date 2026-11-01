"""Command line entry point for repeatable benchmark runs."""

import argparse
import sys
from itertools import product
from pathlib import Path

import torch

from .data import load_dataset
from .experiment import RunConfig, run_experiment
from .reporting import run_label, save_environment, save_plot, save_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["synthetic", "a9a", "fashion_mnist", "cifar10"],
                        default="synthetic")
    parser.add_argument("--optimizers", nargs="+", choices=["custom", "torch"],
                        default=["custom"])
    parser.add_argument("--samplers", nargs="+", choices=["reshuffle", "once", "sequential"],
                        default=["reshuffle", "once"])
    parser.add_argument("--schedulers", nargs="+", choices=["none", "cosine", "exponential", "plateau"],
                        default=["none"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 128])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--train-limit", type=int, default=None,
                        help="Use the same first N training samples in each run (quick experiments)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/latest"))
    parser.add_argument("--download", action="store_true",
                        help="Download a9a or torchvision data if missing")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    bundle = load_dataset(args.dataset, args.data_dir, args.seed, args.download)
    combinations = list(product(args.optimizers, args.samplers,
                                args.schedulers, args.batch_sizes))
    if len(combinations) > 32:
        raise ValueError("At most 32 configurations per invocation; narrow the grid")
    results = []
    for index, (optimizer, sampler, scheduler, batch_size) in enumerate(combinations, 1):
        config = RunConfig(dataset=args.dataset, optimizer=optimizer, sampler=sampler,
                           scheduler=scheduler, batch_size=batch_size, epochs=args.epochs,
                           lr=args.lr, momentum=args.momentum,
                           weight_decay=args.weight_decay, seed=args.seed, device=device,
                           train_limit=args.train_limit)
        print(f"[{index}/{len(combinations)}] {optimizer}/{sampler}/{scheduler}/batch={batch_size}",
              flush=True)
        result = run_experiment(bundle, config)
        results.append(result)
        print(f"  best epoch {result['best_epoch']}; "
              f"test accuracy {result['test']['accuracy']:.3f}; "
              f"F1 {result['test']['f1']:.3f}; "
              f"test loss {result['test']['loss']:.4f}", flush=True)
    save_results(results, args.output_dir)
    save_environment(args.output_dir, args.dataset, args.data_dir,
                     ["stochopt-bench", *(argv if argv is not None else sys.argv[1:])])
    if not args.no_plot:
        save_plot(results, args.output_dir)
    print(f"Saved {len(results)} runs to {args.output_dir.resolve()}")
    best = min(results, key=lambda row: row["best_validation_loss"])
    print(f"Lowest validation loss: {run_label(best)} "
          f"({best['best_validation_loss']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
