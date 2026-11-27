"""
feature_pipeline.py - Class 5 Capstone
General + Industrial feature pipeline
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np

def _write_quality(path, rows):
    pd.DataFrame(rows, columns=["check", "value"]).to_csv(path, index=False)

def run_general(measurements_csv, site_metadata_json, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(measurements_csv)
    df.columns = [c.strip() for c in df.columns]

    with open(site_metadata_json) as f:
        meta = json.load(f)
    site_to_region = {s["lab_site"]: s["region"] for s in meta.get("sites", [])}

    n_in = len(df)
    df["region"] = df["lab_site"].map(site_to_region)

    meas_cols = ["mean_radius", "mean_texture", "mean_perimeter", "mean_area"]
    missing_mask = df[meas_cols].isna().any(axis=1)
    invalid_mask = (
        (df["mean_area"] <= 0)
        | (df["mean_texture"] > 50)
        | (df["mean_radius"] < 5) | (df["mean_radius"] > 30)
        | (df["mean_perimeter"] < 30) | (df["mean_perimeter"] > 250)
    )
    unmatched_mask = df["region"].isna()
    quarantine_mask = missing_mask | invalid_mask | unmatched_mask

    clean = df[~quarantine_mask].copy()

    clean["area_per_perimeter"] = clean["mean_area"] / clean["mean_perimeter"]
    clean["radius_texture_ratio"] = clean["mean_radius"] / clean["mean_texture"]
    clean["log_area"] = np.log1p(clean["mean_area"])
    clean["perimeter_radius_ratio"] = clean["mean_perimeter"] / clean["mean_radius"]

    train = clean[clean["lab_site"].isin(["NORTH", "EAST"])].copy()
    test = clean[clean["lab_site"].isin(["WEST"])].copy()

    train.to_csv(output_dir / "general_train_features.csv", index=False)
    test.to_csv(output_dir / "general_test_features.csv", index=False)

    _write_quality(output_dir / "general_quality_report.csv", [
        ("n_input_rows", n_in),
        ("n_clean_rows", len(clean)),
        ("n_quarantined_rows", int(quarantine_mask.sum())),
        ("n_train_rows", len(train)),
        ("n_test_rows", len(test)),
    ])

    schema = {
        "train_test_split": "Train=NORTH+EAST, Test=WEST (held-out site), SOUTH quarantined (no metadata)",
        "columns": {c: str(clean[c].dtype) for c in clean.columns},
        "features_added": ["area_per_perimeter", "radius_texture_ratio", "log_area", "perimeter_radius_ratio"],
    }
    with open(output_dir / "general_feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    return train, test

def _parse_las(path):
    lines = Path(path).read_text().splitlines()
    in_ascii = False
    rows = []
    for line in lines:
        s = line.strip()
        if s.startswith("~A"):
            in_ascii = True
            continue
        if in_ascii:
            if not s or s.startswith("~"):
                continue
            parts = s.split()
            if len(parts) >= 4:
                rows.append([float(x) for x in parts[:4]])
    las = pd.DataFrame(rows, columns=["DEPT", "GR", "RESD", "TEMP"])
    las = las.replace(-999.25, np.nan)
    las = las.rename(columns={"DEPT": "depth_m", "GR": "gr_api", "RESD": "resd_ohmm", "TEMP": "temp_las_degc"})
    return las

def _find_col(df, *keywords):
    for c in df.columns:
        cl = c.lower()
        if all(k in cl for k in keywords):
            return c
    return None

def run_industrial(well_log_las, scada_csv, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    las = _parse_las(well_log_las)
    scada = pd.read_csv(scada_csv)
    scada.columns = [c.strip() for c in scada.columns]

    depth_col = _find_col(scada, "depth")
    press_col = _find_col(scada, "pressure")
    temp_col = _find_col(scada, "temperat")
    flow_col = _find_col(scada, "flow")

    scada = scada.rename(columns={
        depth_col: "depth_m",
        press_col: "pressure_psi",
        temp_col: "temperature_f",
        flow_col: "flow_bpd",
    })

    n_scada_in = len(scada)
    n_las_in = len(las)

    scada["pressure_kpa"] = scada["pressure_psi"] * 6.89476
    scada["temperature_c_scada"] = (scada["temperature_f"] - 32) * 5 / 9

    scada_agg = scada.groupby("depth_m", as_index=False).agg(
        pressure_psi=("pressure_psi", "mean"),
        temperature_f=("temperature_f", "mean"),
        flow_bpd=("flow_bpd", "mean"),
        pressure_kpa=("pressure_kpa", "mean"),
        temperature_c_scada=("temperature_c_scada", "mean"),
        n_scada_records=("depth_m", "size"),
    )

    merged = pd.merge(las, scada_agg, on="depth_m", how="inner")
    unmatched_scada = scada_agg[~scada_agg["depth_m"].isin(las["depth_m"])]

    merged["gr_resd_ratio"] = merged["gr_api"] / merged["resd_ohmm"]
    merged["temp_diff_c"] = merged["temperature_c_scada"] - merged["temp_las_degc"]
    merged["pressure_per_flow"] = merged["pressure_kpa"] / merged["flow_bpd"]
    merged = merged.sort_values("depth_m")
    merged.to_csv(output_dir / "industrial_features.csv", index=False)

    _write_quality(output_dir / "industrial_quality_report.csv", [
        ("n_las_rows", n_las_in),
        ("n_scada_rows_input", n_scada_in),
        ("n_scada_unique_depths", len(scada_agg)),
        ("n_joined_rows", len(merged)),
        ("n_unmatched_scada_depths", len(unmatched_scada)),
        ("unmatched_scada_depths", ",".join(map(str, unmatched_scada["depth_m"].tolist()))),
    ])
    return merged