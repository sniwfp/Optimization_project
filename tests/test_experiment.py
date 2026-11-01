import tempfile
import unittest
from pathlib import Path

import torch

from stochopt.data import load_dataset
from stochopt.experiment import RunConfig, run_experiment
from stochopt.reporting import save_results


class ExperimentTests(unittest.TestCase):
    def test_synthetic_run_is_reproducible_and_writes_summary(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            bundle = load_dataset("synthetic", path, seed=11)
            config = RunConfig(dataset="synthetic", epochs=2, batch_size=64,
                               train_limit=256, seed=11, scheduler="plateau")
            first = run_experiment(bundle, config)
            second = run_experiment(bundle, config)
            self.assertEqual(first["history"], second["history"])
            self.assertEqual(first["test"], second["test"])
            self.assertEqual(len(first["history"]), 2)
            self.assertEqual(first["sizes"]["train"], 256)
            self.assertTrue(0 <= first["test"]["accuracy"] <= 1)
            self.assertTrue(0 <= first["test"]["f1"] <= 1)
            self.assertTrue(0 <= first["test"]["roc_auc"] <= 1)
            save_results([first], path)
            self.assertTrue((path / "results.json").exists())
            self.assertTrue((path / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
