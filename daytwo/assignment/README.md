# Outlier Audit

## Overview
This project audits a dataset for outliers and separates clean rows from suspicious rows, instead of silently deleting them.

Outliers are detected using the IQR method. Rows flagged as outliers are quarantined for review, not deleted.

## Files
- `outlier_audit.py` - Main script. Contains `audit_outliers()` / `audit_data()` which returns retained and quarantined rows.
- `test_outlier_audit.py` - Unit test. Checks that normal rows are retained and extreme rows are quarantined.
- `cleaned_data.csv` - Retained / clean data after audit.
- `quarantined_data.csv` - Rows flagged as outliers for review.

## How It Works
1. Load the dataset into a DataFrame.
2. For each numeric column, calculate Q1, Q3, and IQR.
3. Flag a row as an outlier if it falls outside:
   `Q1 - 1.5 * IQR` to `Q3 + 1.5 * IQR`
4. Return two DataFrames:
   - `retained` - Normal rows, safe to use.
   - `quarantined` - Outlier rows, kept for manual review.

## How To Run
```bash
python outlier_audit.py