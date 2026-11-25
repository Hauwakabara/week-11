import pandas as pd
import numpy as np

CHECK_COLS = ["proline", "malic_acid"]

def audit_outliers(data: pd.DataFrame):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a DataFrame")
    out = data.copy()
    out["domain_invalid"] = (out["proline"] < 0) | (out["malic_acid"] < 0)

    for col in CHECK_COLS:
        s = out[col]
        mean = s.mean()
        std = s.std()
        out[col + "_z"] = (s - mean) / std
        out[col + "_z_flag"] = out[col + "_z"].abs() > 3
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        out[col + "_iqr_flag"] = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)

    quarantined = out[out["domain_invalid"]]
    retained = out[~out["domain_invalid"]]

    rows = []
    for col in CHECK_COLS:
        row = {}
        row["measurement"] = col
        row["total_input"] = int(len(out))
        row["retained"] = int(len(retained))
        row["quarantined"] = int(len(quarantined))
        row["z_flags"] = int(out[col + "_z_flag"].sum())
        row["iqr_flags"] = int(out[col + "_iqr_flag"].sum())
        rows.append(row)

    summary = pd.DataFrame(rows)
    return {"retained": retained, "quarantined": quarantined, "summary": summary}

if __name__ == "__main__":
    df = pd.read_csv("wine_quality_case_PUBLIC_DERIVED.csv")
    result = audit_outliers(df)
    result["retained"].to_csv("retained_flagged.csv", index=False)
    result["quarantined"].to_csv("quarantined_records.csv", index=False)
    result["summary"].to_csv("quality_summary.csv", index=False)
    print(result["summary"])
    print("retained:", len(result["retained"]), "quarantined:", len(result["quarantined"]))
    
    audit_data = audit_outliers