# config/config.py
from pathlib import Path
import pretty_errors
import mlflow
import logging
from rich.logging import RichHandler

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")

# Create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Assets
SUPERSTORE_DATA_URL = 'superstore_data.csv'

# Data Config
CONTINUOUS_COLS = ['Income', 'Kidhome', 'Teenhome', 'Recency', 
              'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
              'MntSweetProducts','MntGoldProds', 'NumDealsPurchases', 
              'NumWebPurchases','NumCatalogPurchases', 'NumStorePurchases', 
              'NumWebVisitsMonth','Age', 'Enrollment']
CATEGORICAL_COLS = ['Complain', 'Education', 'Marital_Status']
CLASSES = ['Rejected', 'Accepted']

# MLFLOW
STORES_DIR = Path(BASE_DIR, "stores")
MODEL_REGISTRY = Path(STORES_DIR, "models")
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))

# Logging
LOGS_DIR = Path(BASE_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging_config = Path(CONFIG_DIR, "logging.config")
logging.config.dictConfig(logging_config)
logger = logging.getLogger()
logger.handlers[0] = RichHandler(markup=True)  # pretty formatting

