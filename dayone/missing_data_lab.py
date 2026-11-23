"""Week 11 Class 1 — independent missing-data lab.

Run from this folder:
    python missing_data_lab.py
"""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).with_name("gapminder_instructional_missing.csv")
OUTPUT = Path(__file__).with_name("outputs")
REQUIRED = {"country", "continent", "year", "lifeExp", "pop", "gdpPercap"}

def load_and_validate(path: Path) -> pd.DataFrame:
    """Read source, check schema, unique country/year and positive population."""
    df = pd.read_csv(path)
    missing_cols = REQUIRED - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    if df.duplicated(subset=["country", "year"]).any():
        raise ValueError("Duplicate country/year rows found.")
    if (df["pop"] <= 0).any():
        raise ValueError("Population must be positive.")
    return df

def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return column, missing_count, missing_pct for every source column."""
    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_pct": (df.isna().sum().values / len(df) * 100).round(2)
    })
    return report

def impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve original observed values; mark and fill only specified missing cells."""
    out = df.copy()
    # Flags BEFORE filling
    out["lifeExp_was_missing"] = out["lifeExp"].isna()
    out["gdpPercap_was_missing"] = out["gdpPercap"].isna()

    # Continent: from same country
    continent_map = out.dropna(subset=["continent"]).drop_duplicates("country").set_index("country")["continent"]
    out["continent"] = out["continent"].fillna(out["country"].map(continent_map))

    # lifeExp + gdpPercap: continent median, fallback global median
    for col in ["lifeExp", "gdpPercap"]:
        global_median = out[col].median()
        continent_median = out.groupby("continent")[col].transform("median")
        out[col] = out[col].fillna(continent_median)
        out[col] = out[col].fillna(global_median)
    return out

def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    source = load_and_validate(DATA)
    report = missingness_report(source)
    cleaned = impute_data(source)
    for col in ["continent", "lifeExp", "gdpPercap"]:
        if cleaned[col].isna().any():
            raise ValueError(f"Still have missing values in {col}")
    if len(cleaned) != len(source):
        raise ValueError(f"Row count changed: {len(source)} -> {len(cleaned)}")
    for col in ["continent", "lifeExp", "gdpPercap"]:
        mask = source[col].notna()
        if not (cleaned.loc[mask, col] == source.loc[mask, col]).all():
            raise ValueError(f"Original observed values changed in {col}")
    report.to_csv(OUTPUT / "missingness_report.csv", index=False)
    cleaned.to_csv(OUTPUT / "imputed_gapminder.csv", index=False)
    print("Created reports in", OUTPUT)
    print(report)
if __name__ == "__main__":
    main()