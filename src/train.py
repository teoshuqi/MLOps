# tagifai/train.py
import json
from argparse import Namespace
from typing import Dict

import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression

from config import config
from src import data, evaluate, predict, utils


def train(args: Namespace, df: pd.DataFrame, trial: optuna.trial._trial.Trial = None) -> Dict:
    """
    Train model on data.

    Args:
        args (Namespace): arguments to use for training
        df (pd.DataFrame): data for training
        trial (optuna.trial._trial.Trial, optional): optimization trial

    Returns:
        Dict: artificats from the run
    """
    utils.set_seeds()

    # Preprocess Data
    cols = config.CONTINUOUS_COLS + config.CATEGORICAL_COLS
    (X, y), label_encoder_store, minmax_scaler_store = data.preprocess(df, cols, "Response")
    selected_columns = [
        "NumCatalogPurchases",
        "NumStorePurchases",
        "MntWines",
        "MntMeatProducts",
        "Recency",
        "Age",
        "Income",
        "NumWebVisitsMonth",
        "Education",
        "MntGoldProds",
        "MntSweetProducts",
    ]
    X_selected = X.loc[:, selected_columns]
    X_train, X_val, X_test, y_train, y_val, y_test = data.get_data_splits(X_selected, y)

    # Oversample minority class (training set)
    oversample = RandomOverSampler(sampling_strategy="all", random_state=42)
    X_over, y_over = oversample.fit_resample(X_train, y_train)

    # Train
    lg_model = LogisticRegression(
        C=args.C,
        solver=args.solver,
        max_iter=args.max_iter,
        tol=args.tol,
        n_jobs=3,
        random_state=42,
    )
    lg_model.fit(X_over, y_over)
    y_val_pred = lg_model.predict(X_val)
    y_val_prob = lg_model.predict_proba(X_val)
    args.threshold = np.quantile(
        [y_val_prob[i][1] for i in range(len(y_val_pred)) if y_val.values[i] == 1], q=0.2
    )  # Q1

    # Evaluation
    y_prob = lg_model.predict_proba(X_test)
    y_pred = predict.custom_predict(y_prob=y_prob, threshold=args.threshold)
    lg_performance = evaluate.get_metrics(y_pred, y_test)
    config.logger.info(json.dumps(lg_performance, indent=2))

    return {
        "args": args,
        "model": lg_model,
        "performance": lg_performance,
        "label_encoder": label_encoder_store,
        "minmax_scaler": minmax_scaler_store,
    }


def objective(args: Namespace, df: pd.DataFrame, trial: optuna.trial._trial.Trial) -> float:
    """
    Objective function for optimization trials.

    Args:
        args (Namespace): arguments to use for training
        df (pd.DataFrame): data for training
        trial (optuna.trial._trial.Trial, optional): optimization trial

    Returns:
        float: metric value (recall) to be used for optimization
    """
    # Parameters to tune
    args.solver = trial.suggest_categorical(
        "solver", ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
    )
    args.max_iter = trial.suggest_int("max_iter", 50, 1000)
    args.C = trial.suggest_uniform("C", 1e-4, 50)
    args.tol = trial.suggest_uniform("tol", 0.00001, 0.1)

    # Train & evaluate
    artifacts = train(args=args, df=df, trial=trial)

    # Set additional attributes
    overall_performance = artifacts["performance"]["overall"]
    accepted_performance = artifacts["performance"]["class"]["Accepted"]
    config.logger.info(json.dumps(overall_performance, indent=2))
    trial.set_user_attr("precision", overall_performance["precision"])
    trial.set_user_attr("recall", overall_performance["recall"])
    trial.set_user_attr("f1", overall_performance["f1"])
    trial.set_user_attr("accepted precision", accepted_performance["precision"])
    trial.set_user_attr("accepted recall", accepted_performance["recall"])
    trial.set_user_attr("accepted f1", accepted_performance["f1"])
    return accepted_performance["recall"]
