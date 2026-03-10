#!/usr/bin/env python3

"""
==============================================================
USAGE
==============================================================

python3 regression_pipeline.py --input data.csv --target y

Optional:
--cores 4
--shap

Example:

python3 regression_pipeline.py \
--input housing.csv \
--target price \
--cores 4 \
--shap

Default CPU threads = 1
If user specifies threads → pipeline adapts.
==============================================================
"""

import argparse
import pandas as pd
import numpy as np
import warnings
import sys

warnings.filterwarnings("ignore")

if sys.version_info < (3,8):
    sys.exit("Python 3.8 or higher required")

from itertools import combinations

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR


# ============================================================
# TERMINAL DISPLAY
# ============================================================

def banner(text):
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80)

def sub_banner(text):
    print("\n" + "-"*80)
    print(text)
    print("-"*80)


# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="Input CSV file")

    parser.add_argument("--target", required=True, help="Target column")

    parser.add_argument("--cores", type=int, default=1,
                        help="CPU threads (default=1)")

    parser.add_argument("--shap", action="store_true",
                        help="Enable SHAP feature importance")

    return parser.parse_args()


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(path):

    banner("LOADING DATASET")

    df = pd.read_csv(path)

    print("Dataset shape:", df.shape)
    print("Total columns:", len(df.columns))

    return df


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

def statistical_analysis(df):

    banner("STATISTICAL ANALYSIS")

    numeric = df.select_dtypes(include=np.number)
    categorical = df.select_dtypes(exclude=np.number)

    if len(numeric.columns) > 0:

        sub_banner("NUMERIC FEATURES")

        stats = numeric.describe().T

        stats["median"] = numeric.median()
        stats["variance"] = numeric.var()

        print(stats)

    if len(categorical.columns) > 0:

        sub_banner("CATEGORICAL FEATURES")

        for col in categorical.columns:

            print("\nColumn:", col)

            print("Unique:", df[col].nunique())

            print("Mode:", df[col].mode()[0])

            print(df[col].value_counts().head())


# ============================================================
# MISSING VALUES
# ============================================================

def check_missing(df):

    banner("MISSING VALUE ANALYSIS")

    miss = df.isnull().sum()

    if miss.sum() == 0:
        print("No missing values detected")
    else:
        print(miss[miss > 0])


# ============================================================
# OUTLIERS
# ============================================================

def detect_outliers(df):

    banner("OUTLIER DETECTION")

    numeric = df.select_dtypes(include=np.number)

    for col in numeric.columns:

        Q1 = numeric[col].quantile(0.25)
        Q3 = numeric[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR

        outliers = numeric[(numeric[col]<lower)|(numeric[col]>upper)][col]

        print(f"\n{col} → {len(outliers)} outliers")


# ============================================================
# DUPLICATE FEATURES
# ============================================================

def duplicate_columns(df):

    banner("DUPLICATE FEATURE DETECTION")

    duplicates = set()

    for col1,col2 in combinations(df.columns,2):

        if df[col1].equals(df[col2]):
            duplicates.add(col2)

    if len(duplicates)==0:

        print("No duplicate columns")

    else:

        print("Duplicates:",duplicates)

        df=df.drop(columns=list(duplicates))

    return df


# ============================================================
# VARIANCE FILTER
# ============================================================

def variance_filter(df,threshold=0.0):

    banner("VARIANCE FILTER")

    numeric_df = df.select_dtypes(include=np.number)

    selector = VarianceThreshold(threshold)

    selector.fit(numeric_df)

    kept = numeric_df.columns[selector.get_support()]

    removed = set(numeric_df.columns) - set(kept)

    print("Removed:",removed)

    return numeric_df[kept]


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def correlation_analysis(df):

    banner("CORRELATION ANALYSIS")

    corr = df.select_dtypes(include=np.number).corr().abs()

    high=[]

    for c1,c2 in combinations(corr.columns,2):

        if corr.loc[c1,c2]>0.9:

            high.append((c1,c2,corr.loc[c1,c2]))

    if len(high)==0:

        print("No high correlations")

    else:

        for h in high:
            print(h)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(X):

    banner("NORMALIZATION")

    scaler=StandardScaler()

    X_scaled=scaler.fit_transform(X)

    X_scaled=pd.DataFrame(X_scaled,columns=X.columns)

    print("Normalization completed (mean≈0, std≈1)")

    return X_scaled


# ============================================================
# MODELS
# ============================================================

def get_models(cores):

    return {

        "LinearRegression":LinearRegression(),

        "Ridge":Ridge(),

        "Lasso":Lasso(),

        "RandomForest":RandomForestRegressor(
            n_estimators=200,
            n_jobs=cores,
            random_state=42
        ),

        "ExtraTrees":ExtraTreesRegressor(
            n_estimators=200,
            n_jobs=cores,
            random_state=42
        ),

        "SVR":SVR()
    }


# ============================================================
# CROSS VALIDATED METRICS
# ============================================================

def regression_metrics(model,X,y,cores):

    kf=KFold(n_splits=5,shuffle=True,random_state=42)

    r2=cross_val_score(model,X,y,cv=kf,scoring="r2",n_jobs=cores).mean()

    rmse=(-cross_val_score(model,X,y,cv=kf,
          scoring="neg_root_mean_squared_error",
          n_jobs=cores).mean())

    mae=(-cross_val_score(model,X,y,cv=kf,
          scoring="neg_mean_absolute_error",
          n_jobs=cores).mean())

    return r2,rmse,mae


# ============================================================
# FEATURE STABILITY
# ============================================================

def feature_stability(model,X,y):

    banner("FEATURE STABILITY SCORE")

    kf=KFold(n_splits=5,shuffle=True,random_state=42)

    rankings=[]

    for train,test in kf.split(X):

        selector=RFE(model,n_features_to_select=1)

        selector.fit(X.iloc[train],y.iloc[train])

        rankings.append(selector.ranking_)

    rankings=np.array(rankings)

    stability=np.std(rankings,axis=0)

    stability_df=pd.DataFrame({

        "Feature":X.columns,
        "StabilityScore":stability

    }).sort_values("StabilityScore")

    print(stability_df.head(20))


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

def permutation_importance_analysis(model,X,y):

    banner("PERMUTATION FEATURE IMPORTANCE")

    model.fit(X,y)

    result=permutation_importance(model,X,y,n_repeats=10)

    imp=pd.DataFrame({

        "Feature":X.columns,
        "Importance":result.importances_mean

    }).sort_values("Importance",ascending=False)

    print(imp.head(20))


# ============================================================
# SHAP IMPORTANCE
# ============================================================

def shap_analysis(model,X,y,enable):

    if not enable:
        return

    try:

        import shap

        banner("SHAP FEATURE IMPORTANCE")

        model.fit(X,y)

        explainer=shap.Explainer(model,X)

        shap_values=explainer(X[:100])

        shap_mean=np.abs(shap_values.values).mean(axis=0)

        shap_df=pd.DataFrame({

            "Feature":X.columns,
            "SHAP_Importance":shap_mean

        }).sort_values("SHAP_Importance",ascending=False)

        print(shap_df.head(20))

    except ImportError:

        print("\nSHAP not installed → skipping")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    args=parse_arguments()

    if args.cores < 1:
        print("ERROR: cores must be >= 1")
        sys.exit(1)

    df=load_dataset(args.input)

    if args.target not in df.columns:
        print(f"ERROR: Target column '{args.target}' not found.")
        sys.exit(1)

    statistical_analysis(df.drop(columns=[args.target], errors="ignore"))

    check_missing(df)

    detect_outliers(df)

    df=duplicate_columns(df)

    y=df[args.target]

    X=df.drop(columns=[args.target])

    X=variance_filter(X)

    correlation_analysis(X)

    X=normalize(X)

    models=get_models(args.cores)

    banner("MODEL EVALUATION")

    for name,model in models.items():

        sub_banner(name)

        r2,rmse,mae=regression_metrics(model,X,y,args.cores)

        print("R2 :",round(r2,4))
        print("RMSE :",round(rmse,4))
        print("MAE :",round(mae,4))

        feature_stability(model,X,y)

        permutation_importance_analysis(model,X,y)

        shap_analysis(model,X,y,args.shap)

    banner("FEATUREFLOW REGRESSION PIPELINE COMPLETED")


if __name__=="__main__":

    main()
