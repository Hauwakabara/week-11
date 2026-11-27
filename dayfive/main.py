from pathlib import Path
from src.feature_pipeline import run_general, run_industrial

BASE = Path(__file__).parent
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)

print("Running GENERAL...")
train, test = run_general(
    BASE / "data/general_measurements_PUBLIC_DERIVED.csv",
    BASE / "data/site_metadata_SYNTHETIC.json",
    OUT
)
print(f"general: train={len(train)} test={len(test)}")

print("Running INDUSTRIAL...")
merged = run_industrial(
    BASE / "data/well_log.las",
    BASE / "data/scada.csv",
    OUT
)
print(f"industrial: joined={len(merged)}")
print("DONE - check output/ folder")