from pathlib import Path
from src.feature_pipeline import run_general, run_industrial

def test_general(tmp_path):
    train, test = run_general(
        "data/general_measurements_PUBLIC_DERIVED.csv",
        "data/site_metadata_SYNTHETIC.json",
        tmp_path
    )
    assert len(train) > 0
    assert len(test) > 0
    assert "area_per_perimeter" in train.columns
    assert set(test["lab_site"].unique()) == {"WEST"}

def test_industrial(tmp_path):
    merged = run_industrial("data/well_log.las", "data/scada.csv", tmp_path)
    assert len(merged) > 0
    assert "gr_resd_ratio" in merged.columns
    assert "temp_diff_c" in merged.columns