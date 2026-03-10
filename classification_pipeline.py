#!/usr/bin/env python3

"""
Usage:
python3 classification_pipeline.py --input file.csv --target label --cores x

If cores not provided → default = 1
"""

import argparse
import pandas as pd
import numpy as np
import warnings
import sys
warnings.filterwarnings("ignore")

if sys.version_info < (3,8):
    sys.exit("Python 3.8 or higher required")

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC

from itertools import combinations


# =========================================================
# TERMINAL DISPLAY UTILITIES
# =========================================================

def banner(text):
    print("\n" + "="*80)
    print(f"{text.center(80)}")
    print("="*80)

def sub_banner(text):
    print("\n" + "-"*80)
    print(text)
    print("-"*80)


# =========================================================
# ARGUMENT PARSER
# =========================================================

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV dataset"
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Target label column"
    )

    parser.add_argument(
        "--cores",
        required=False,
        type=int,
        default=1,
        help="CPU cores (default=1)"
    )

    return parser.parse_args()


# =========================================================
# LOAD DATA
# =========================================================

def load_dataset(file):

    banner("LOADING DATASET")

    df = pd.read_csv(file)

    print("Dataset Shape:", df.shape)
    print("Columns:", len(df.columns))

    return df


# =========================================================
# STATISTICAL ANALYSIS
# =========================================================

def statistical_analysis(df):

    banner("STATISTICAL ANALYSIS")

    numeric_cols = df.select_dtypes(include=np.number)
    categorical_cols = df.select_dtypes(exclude=np.number)

    if numeric_cols.shape[1] > 0:

        sub_banner("NUMERIC FEATURE STATISTICS")

        desc = numeric_cols.describe(percentiles=[0.25,0.5,0.75]).T

        desc["median"] = numeric_cols.median()
        desc["mode"] = numeric_cols.mode().iloc[0]
        desc["variance"] = numeric_cols.var()

        print(desc)

    else:
        print("No numeric columns detected")

    if categorical_cols.shape[1] > 0:

        sub_banner("CATEGORICAL FEATURE ANALYSIS")

        for col in categorical_cols.columns:

            print(f"\nColumn: {col}")
            print("Unique Values:", categorical_cols[col].nunique())
            print("Mode:", categorical_cols[col].mode()[0])

            print("Top Frequencies:")
            print(categorical_cols[col].value_counts().head(10))

    else:
        print("No categorical columns detected")


# =========================================================
# MISSING VALUES
# =========================================================

def check_missing(df):

    banner("MISSING VALUE ANALYSIS")

    missing = df.isnull().sum()

    if missing.sum() == 0:
        print("No Missing Values Detected")
    else:
        print(missing[missing > 0])


# =========================================================
# OUTLIER DETECTION
# =========================================================

def detect_outliers(df):

    banner("OUTLIER DETECTION (IQR METHOD)")

    numeric = df.select_dtypes(include=np.number)

    for col in numeric.columns:

        Q1 = numeric[col].quantile(0.25)
        Q3 = numeric[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = numeric[(numeric[col] < lower) | (numeric[col] > upper)][col]

        print(f"\nColumn: {col}")
        print("Outlier Count:", len(outliers))

        if len(outliers) > 0:
            print("Outlier Values:", outliers.values[:20])


# =========================================================
# DUPLICATE FEATURES
# =========================================================

def duplicate_columns(df):

    banner("DUPLICATE COLUMN DETECTION")

    duplicates = set()

    for col1, col2 in combinations(df.columns, 2):

        if df[col1].equals(df[col2]):
            duplicates.add(col2)

    if len(duplicates) == 0:
        print("No duplicate columns")
    else:
        print("Duplicate columns:", duplicates)

    df = df.drop(columns=list(duplicates))

    return df


# =========================================================
# VARIANCE FILTER
# =========================================================

def variance_filter(df, threshold=0.1):

    banner("VARIANCE THRESHOLD FILTERING")

    numeric_df = df.select_dtypes(include=np.number)

    variances = numeric_df.var()

    print("\nFeature Variances:")
    print(variances)

    selector = VarianceThreshold(threshold)

    try:

        selector.fit(numeric_df)

        kept_columns = numeric_df.columns[selector.get_support()]

        removed = set(numeric_df.columns) - set(kept_columns)

        print("\nRemoved Low Variance Columns:", removed)

        return numeric_df[kept_columns]

    except ValueError:

        print("\nWARNING:")
        print("No features meet the variance threshold =", threshold)

        print("\nSwitching to adaptive threshold (0.0)")

        selector = VarianceThreshold(0.0)

        selector.fit(numeric_df)

        kept_columns = numeric_df.columns[selector.get_support()]

        return numeric_df[kept_columns]


# =========================================================
# CORRELATION ANALYSIS
# =========================================================

def correlation_analysis(df):

    banner("CORRELATION ANALYSIS")

    corr = df.select_dtypes(include=np.number).corr().abs()

    pairs = []

    for col1, col2 in combinations(corr.columns, 2):

        val = corr.loc[col1, col2]

        if val > 0.9:
            pairs.append((col1,col2,val))

    if len(pairs) == 0:
        print("No high correlations")

    else:
        print("Highly correlated pairs (>0.9):")

        for p in pairs:
            print(p)


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_data(X):

    banner("NORMALIZATION (STANDARD SCALER)")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    print("Normalization completed successfully.\n")

    stats = pd.DataFrame({
        "Mean": X_scaled.mean(),
        "StdDev": X_scaled.std()
    })

    print(stats)

    print("\nSample of normalized data:")
    print(X_scaled.head())

    return X_scaled


# =========================================================
# MODEL DEFINITIONS
# =========================================================

def get_models(cores):

    models = {

        "LogisticRegression": LogisticRegression(
            max_iter=200,
            n_jobs=cores,
            random_state=42
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            n_jobs=cores,
            random_state=42
        ),

        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=100,
            n_jobs=cores,
            random_state=42
        ),

        "LinearSVM": LinearSVC(max_iter=5000)

    }

    return models


# =========================================================
# INCREMENTAL FEATURE SELECTION
# =========================================================

def incremental_feature_selection(X, y, models):

    banner("INCREMENTAL FEATURE SELECTION (IFS)")

    features = list(X.columns)

    results = {}

    for name, model in models.items():

        sub_banner(f"MODEL: {name}")

        scores = {}

        for i in range(1, len(features)+1):

            subset = features[:i]

            score = cross_val_score(
                model,
                X[subset],
                y,
                cv=5,
                scoring="accuracy"
            ).mean()

            scores[tuple(subset)] = score

        ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)

        for f,s in ranked[:10]:

            print(f"Features: {list(f)} | Relevance: {round(s*100,2)} %")

        results[name] = ranked

    return results


# =========================================================
# RECURSIVE FEATURE ELIMINATION
# =========================================================

def recursive_feature_elimination(X, y, models):

    banner("RECURSIVE FEATURE ELIMINATION (RFE)")

    for name, model in models.items():

        sub_banner(f"MODEL: {name}")

        selector = RFE(model, n_features_to_select=1)

        selector.fit(X, y)

        ranking = selector.ranking_

        importance = 1 / ranking

        importance = importance / importance.sum()

        ranking_df = pd.DataFrame({

            "Feature": X.columns,
            "Relevance": importance*100

        })

        ranking_df = ranking_df.sort_values(
            "Relevance",
            ascending=False
        )

        print(ranking_df)


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    args = parse_arguments()

    if args.cores < 1:
        print("ERROR: cores must be >= 1")
        sys.exit(1)

    df = load_dataset(args.input)

    if args.target not in df.columns:
        print(f"ERROR: Target column '{args.target}' not found in dataset.")
        sys.exit(1)

    statistical_analysis(df.drop(columns=[args.target], errors="ignore"))

    check_missing(df)

    detect_outliers(df)

    df = duplicate_columns(df)

    target = args.target

    y = df[target]

    X = df.drop(columns=[target])

    X = variance_filter(X)

    correlation_analysis(X)

    X = normalize_data(X)

    models = get_models(args.cores)

    incremental_feature_selection(X, y, models)

    recursive_feature_elimination(X, y, models)

    banner("FEATUREFLOW CLASSIFICATION PIPELINE COMPLETED")


if __name__ == "__main__":
    main()
