from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.pre_proc_pipeline import make_stratified_splits


class SplitContractTests(unittest.TestCase):
    def test_make_stratified_splits_is_deterministic_and_disjoint(self) -> None:
        df = pd.DataFrame(
            {
                "id_code": [f"img_{i:03d}" for i in range(100)],
                "diagnosis": [i % 5 for i in range(100)],
            }
        )

        first = make_stratified_splits(
            df,
            id_column="id_code",
            label_column="diagnosis",
            random_state=42,
            val_size=0.1,
            test_size=0.1,
        )
        second = make_stratified_splits(
            df,
            id_column="id_code",
            label_column="diagnosis",
            random_state=42,
            val_size=0.1,
            test_size=0.1,
        )

        self.assertEqual(
            first["train"]["id_code"].tolist(),
            second["train"]["id_code"].tolist(),
        )
        self.assertEqual(
            first["val"]["id_code"].tolist(),
            second["val"]["id_code"].tolist(),
        )
        self.assertEqual(
            first["test"]["id_code"].tolist(),
            second["test"]["id_code"].tolist(),
        )

        train_ids = set(first["train"]["id_code"])
        val_ids = set(first["val"]["id_code"])
        test_ids = set(first["test"]["id_code"])

        self.assertFalse(train_ids & val_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(val_ids & test_ids)
        self.assertEqual(len(train_ids | val_ids | test_ids), len(df))

    def test_split_sizes_are_80_10_10(self) -> None:
        df = pd.DataFrame(
            {
                "id_code": [f"img_{i:03d}" for i in range(100)],
                "diagnosis": [i % 5 for i in range(100)],
            }
        )

        splits = make_stratified_splits(df)

        self.assertEqual(len(splits["train"]), 80)
        self.assertEqual(len(splits["val"]), 10)
        self.assertEqual(len(splits["test"]), 10)


if __name__ == "__main__":
    unittest.main()
