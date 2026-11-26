import argparse
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="iris_preprocessing_case_PUBLIC_DERIVED.csv")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    # Load
    df = pd.read_csv(args.input)
    source_rows = len(df)
    missing_source_cells = int(df.isna().sum().sum())
    print(f"Source rows: {source_rows}, missing cells: {missing_source_cells}")
    print(df.head())

    # Split X and y - exclude sample_id
    X = df.drop(columns=["species", "sample_id"])
    y = df["species"]

    # Train / test split FIRST - before fitting anything
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"Numeric: {numeric_features}")
    print(f"Categorical: {categorical_features}")

    # Preprocessor: median + scaler for numeric, most_frequent + one-hot for categorical
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])

    # Fit ONLY on train, transform both
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Get feature names
    feature_names = preprocessor.get_feature_names_out()
    print(f"Transformed features: {len(feature_names)}")

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    train_df = pd.DataFrame(X_train_t, columns=feature_names)
    test_df = pd.DataFrame(X_test_t, columns=feature_names)
    train_df.to_csv(os.path.join(args.output_dir, "train_features.csv"), index=False)
    test_df.to_csv(os.path.join(args.output_dir, "test_features.csv"), index=False)

    summary = pd.DataFrame([{
        "source_rows": source_rows,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "transformed_features": len(feature_names),
        "missing_source_cells": missing_source_cells,
    }])
    summary.to_csv(os.path.join(args.output_dir, "data_quality_summary.csv"), index=False)

    print("Saved to outputs/: train_features.csv, test_features.csv, data_quality_summary.csv")
    print("Check finiteness:", train_df.notna().all().all(), test_df.notna().all().all())

if __name__ == "__main__":
    main()