import unittest
import pandas as pd
from outlier_audit import audit_data

def make_df():
    return pd.DataFrame({
        "sample_id": ["W1", "W2", "W3", "W4"],
        "alcohol": [13.0, 13.1, 13.2, 13.3],
        "malic_acid": [1.5, 1.8, 2.0, 99.0],
        "color_intensity": [5.0, 5.1, 5.2, 5.3],
        "proline": [700, 800, 900, -45],
        "target": [0, 0, 1, 1],
    })

class TestOutlierAudit(unittest.TestCase):
    def test_quarantine_only_domain_invalid(self):
        df = make_df()
        result = audit_data(df)
        retained = result["retained"]
        quarantined = result["quarantined"]
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(quarantined.iloc[0]["sample_id"], "W4")
        self.assertIn("W3", retained["sample_id"].tolist())
        # W4 is the outlier (malic 99.0) - flagged AND quarantined
        w4 = quarantined[quarantined["sample_id"] == "W4"].iloc[0]
        self.assertTrue(w4["review_needed"])
        self.assertTrue(w4["malic_acid_iqr_flag"])

    def test_no_flag_for_normal_rows(self):
        df = make_df()
        result = audit_data(df)
        retained = result["retained"]
        w1 = retained[retained["sample_id"] == "W1"].iloc[0]
        self.assertFalse(w1["review_needed"])

    def test_original_values_not_modified(self):
        df = make_df()
        result = audit_data(df)
        retained = result["retained"]
        quarantined = result["quarantined"]
        self.assertEqual(quarantined[quarantined["sample_id"] == "W4"].iloc[0]["malic_acid"], 99.0)
        self.assertEqual(quarantined.iloc[0]["proline"], -45)
        self.assertEqual(retained[retained["sample_id"] == "W1"].iloc[0]["malic_acid"], 1.5)

    def test_zero_variance_column(self):
        df = make_df()
        result = audit_data(df)
        retained = result["retained"]
        quarantined = result["quarantined"]
        self.assertEqual(len(quarantined), 1)

if __name__ == "__main__":
    unittest.main()