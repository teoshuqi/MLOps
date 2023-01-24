# tagifai/main.py
import pandas as pd
from pathlib import Path
from argparse import Namespace
import warnings, json
import mlflow
import joblib
import tempfile
from numpyencoder import NumpyEncoder
import optuna
from optuna.integration.mlflow import MLflowCallback

from config import config
from src import utils, data, train, predict

warnings.filterwarnings("ignore")

def elt_data():
    """Extract, load and transform our data assets."""
    # Extract + Load

    # Transform
    return

def train_model(args_fp, experiment_name, run_name):
    """Train a model given arguments."""
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, config.SUPERSTORE_DATA_URL))

    # Train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        config.logger.info(f"Run ID: {run_id}")
        artifacts = train.train(df=df, args=args)
        performance = artifacts["performance"]
        config.logger.info(json.dumps(performance, indent=2))

        # Log metrics and parameters
        performance = artifacts["performance"]
        mlflow.log_metrics({"precision": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1": performance["overall"]["f1"]})
        mlflow.log_params(vars(artifacts["args"]))

        config.logger.info(artifacts["label_encoder"])
        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            joblib.dump(artifacts["label_encoder"], Path(dp, "label_encoder.pkl"))
            joblib.dump(artifacts["minmax_scaler"], Path(dp, "minmax_scaler.pkl"))
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(performance, Path(dp, "performance.json"))
            utils.save_dict(args.__dict__, Path(dp, "args.json"))
            mlflow.log_artifacts(dp)

    # Save to config
    open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
    utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))


def optimize(args_fp, study_name, num_trials):
    """Optimize hyperparameters."""
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, config.SUPERSTORE_DATA_URL))

    # Optimize
    args = Namespace(**utils.load_dict(filepath=args_fp))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name="optimization2", direction="maximize", pruner=pruner)

    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
    study.optimize(
        lambda trial: train.objective(args, df, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback])

    # Best trial
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["user_attrs_f1"], ascending=False)
    utils.save_dict({**args.__dict__, **study.best_trial.params}, args_fp, cls=NumpyEncoder)
    config.logger.info(f"\nBest value (f1): {study.best_trial.value}")
    config.logger.info(f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")

def load_artifacts(run_id):
    """Load artifacts for a given run_id."""
    # Locate specifics artifacts directory
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")
    config.logger.info(artifacts_dir)
    # Load objects from run
    args = Namespace(**utils.load_dict(filepath=Path(artifacts_dir, "args.json")))
    minmax_scaler = joblib.load(Path(artifacts_dir, "minmax_scaler.pkl"))
    label_encoder = joblib.load(Path(artifacts_dir, "label_encoder.pkl"))
    model = joblib.load(Path(artifacts_dir, "model.pkl"))
    performance = utils.load_dict(filepath=Path(artifacts_dir, "performance.json"))

    return {
        "args": args,
        "label_encoder": label_encoder,
        "minmax_scaler": minmax_scaler,
        "model": model,
        "performance": performance
    }

def predict_acceptance(data, run_id=None):
    """Predict if customers will accept superstore campaign."""
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(customer_data=data, artifacts=artifacts)
    config.logger.info(json.dumps(prediction, indent=2))
    return prediction