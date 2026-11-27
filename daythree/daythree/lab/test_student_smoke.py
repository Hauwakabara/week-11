import os
import unittest
import pandas as pd

class TestPreprocessing(unittest.TestCase):
    def test_outputs_exist_and_match(self):
        base = os.path.dirname(__file__)
        out = os.path.join(base, "outputs")
        # These files should exist after running preprocess_iris.py
        train_path = os.path.join(out, "train_features.csv")
        test_path = os.path.join(out, "test_features.csv")
        summary_path = os.path.join(out, "data_quality_summary.csv")

        self.assertTrue(os.path.exists(train_path), "train_features.csv missing - run preprocess_iris.py first")
        self.assertTrue(os.path.exists(test_path), "test_features.csv missing")
        self.assertTrue(os.path.exists(summary_path), "data_quality_summary.csv missing")

        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

        # 120 train, 30 test
        self.assertEqual(len(train), 120)
        self.assertEqual(len(test), 30)
        # Same columns
        self.assertListEqual(list(train.columns), list(test.columns))
        # No missing / infinite
        self.assertFalse(train.isna().any().any())
        self.assertFalse(test.isna().any().any())

    def test_source_csv_exists(self):
        base = os.path.dirname(__file__)
        path = os.path.join(base, "iris_preprocessing_case_PUBLIC_DERIVED.csv")
        self.assertTrue(os.path.exists(path))

if __name__ == "__main__":
    unittest.main()