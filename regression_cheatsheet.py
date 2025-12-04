"""
Regression CheatSheet - regression_cheatsheet.py
Author: Aditya Bhatt
Date: 2024-06-15

Quick repo starter file for your Regression CheatSheet repo.
Includes runnable examples (using sklearn.datasets.load_diabetes by default),
and functions for:
 - Linear, Ridge, Lasso, ElasticNet
 - Polynomial (pipeline)
 - SVR
 - DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor
 - Pipeline example (StandardScaler + Ridge)
 - Optional placeholders for XGBoost/LightGBM (commented)

Requirements (minimal):
 numpy, pandas, scikit-learn, matplotlib (optional)
 Optional: xgboost, lightgbm (commented)
"""

# Imports
import warnings
warnings.filterwarnings("ignore")

import os
import argparse
from typing import Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Regression models / tools
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import xgboost as xgb

# ---------- Utilities ----------

def load_data(dataset: str = "diabetes", test_size: float = 0.2, random_state: int = 42
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset and split.
    dataset: 'diabetes' or 'custom dataset' (expects data/filename.csv with say 4 columns)
    returns: X_train, X_test, y_train, y_test
    """
    if dataset == "house":
        path = os.path.join("data", "filename.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Place your CSV at this path or use --data diabetes")
        df = pd.read_csv(path)
        X = df[["column1", "column2", "column3"]].values
        y = df["column4"].values
    else:
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = data["data"]
        y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def print_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str = None) -> Dict[str, float]:
    """Compute and print R2, MSE, MAE. Returns a dict of metrics."""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    if name:
        print(f"[{name}] R2: {r2:.4f} | MSE: {mse:.4f} | MAE: {mae:.4f}")
    else:
        print(f"R2: {r2:.4f} | MSE: {mse:.4f} | MAE: {mae:.4f}")
    return {"r2": r2, "mse": mse, "mae": mae}


# ---------- Models & examples ----------

def train_and_eval_linear(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return print_regression_metrics(y_test, y_pred, "LinearRegression")


def train_and_eval_ridge_lasso_enet(X_train, X_test, y_train, y_test):
    results = {}
    ridge = Ridge(alpha=1.0, random_state=42).fit(X_train, y_train)
    results["ridge"] = print_regression_metrics(y_test, ridge.predict(X_test), "Ridge")

    lasso = Lasso(alpha=0.01, random_state=42, max_iter=10000).fit(X_train, y_train)
    results["lasso"] = print_regression_metrics(y_test, lasso.predict(X_test), "Lasso")

    enet = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000).fit(X_train, y_train)
    results["elasticnet"] = print_regression_metrics(y_test, enet.predict(X_test), "ElasticNet")

    return results


def train_and_eval_polynomial(X_train, X_test, y_train, y_test, degree: int = 3):
    pipe = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scale", StandardScaler(with_mean=False)),
            ("lin", Ridge(alpha=1.0)),
        ]
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return print_regression_metrics(y_test, y_pred, f"Polynomial(deg={degree})")


def train_and_eval_svr(X_train, X_test, y_train, y_test):
    svr = SVR(kernel="rbf", C=1.0, epsilon=0.1).fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    return print_regression_metrics(y_test, y_pred, "SVR")


def train_and_eval_tree_ensembles(X_train, X_test, y_train, y_test):
    results = {}
    dt = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
    results["dt"] = print_regression_metrics(y_test, dt.predict(X_test), "DecisionTree")

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train, y_train)
    results["rf"] = print_regression_metrics(y_test, rf.predict(X_test), "RandomForest")

    gbr = GradientBoostingRegressor(learning_rate=0.05, n_estimators=200, max_depth=3, random_state=42).fit(
        X_train, y_train
    )
    results["gbr"] = print_regression_metrics(y_test, gbr.predict(X_test), "GradientBoosting")

    return results


def train_and_eval_xgboost(X_train, X_test, y_train, y_test):
    xg = xgb.XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X_train, y_train)
    return print_regression_metrics(y_test, xg.predict(X_test), "XGBoost")

def pipeline_ridge(X_train, X_test, y_train, y_test):
    pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return print_regression_metrics(y_test, y_pred, "Pipeline-Ridge")


# ---------- Demo runner ----------

def run_all(dataset: str = "diabetes"):
    X_train, X_test, y_train, y_test = load_data(dataset=dataset)
    print("\n=== Regression CheatSheet Demo (dataset: {}) ===".format(dataset))
    train_and_eval_linear(X_train, X_test, y_train, y_test)
    train_and_eval_ridge_lasso_enet(X_train, X_test, y_train, y_test)
    train_and_eval_polynomial(X_train, X_test, y_train, y_test, degree=2)
    train_and_eval_svr(X_train, X_test, y_train, y_test)
    train_and_eval_tree_ensembles(X_train, X_test, y_train, y_test)
    train_and_eval_xgboost(X_train, X_test, y_train, y_test)
    pipeline_ridge(X_train, X_test, y_train, y_test)
    
if __name__ == "__main__":
    run_all()