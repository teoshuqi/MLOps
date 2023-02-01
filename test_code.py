from config import config
from pathlib import Path
from src import main
import pandas as pd

args_fp = Path(config.CONFIG_DIR, "args.json")
main.optimize(args_fp, study_name="optimization", num_trials=100)

args_fp = Path(config.CONFIG_DIR, "args.json")
main.train_model(args_fp, experiment_name="baselines", run_name="lg")

df = pd.read_csv(Path(config.DATA_DIR, config.SUPERSTORE_DATA_URL))
run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
main.predict_acceptance(data=df.iloc[0:1,:], run_id=run_id)