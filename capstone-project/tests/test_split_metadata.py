import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.split_metadata import class_distribution, make_dataset_provided_split_metadata


class SplitMetadataTests(unittest.TestCase):
    def test_class_distribution_uses_stable_string_keys(self):
        distribution = class_distribution(np.array([2, 0, 2, 1, 1, 1]))

        self.assertEqual(distribution, {"0": 1, "1": 3, "2": 2})

    def test_dataset_provided_metadata_records_raw_and_saved_counts(self):
        df = pd.DataFrame(
            {
                "id_code": ["a", "b", "c"],
                "diagnosis": [0, 1, 1],
            }
        )
        y_saved = np.array([0, 1, 1, 1])

        metadata = make_dataset_provided_split_metadata(
            split="train",
            csv_name="train_1.csv",
            image_folder="train_images/train_images",
            df=df,
            y_saved=y_saved,
            id_column="id_code",
            label_column="diagnosis",
            smote_applied_to_saved_arrays=True,
        )

        self.assertEqual(metadata["csv"], "train_1.csv")
        self.assertEqual(metadata["image_folder"], "train_images/train_images")
        self.assertEqual(metadata["raw_samples"], 3)
        self.assertEqual(metadata["raw_class_distribution"], {"0": 1, "1": 2})
        self.assertEqual(metadata["saved_samples"], 4)
        self.assertEqual(metadata["saved_class_distribution"], {"0": 1, "1": 3})
        self.assertTrue(metadata["smote_applied_to_saved_arrays"])


if __name__ == "__main__":
    unittest.main()
