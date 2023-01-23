# config/config.py
from pathlib import Path
import pretty_errors

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")

# Create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Assets
SUPERSTORE_DATA_URL = 'superstore_data.csv'

# Data Config
CONTINUOUS_COL = ['Income', 'Kidhome', 'Teenhome', 'Recency', 
              'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
              'MntSweetProducts','MntGoldProds', 'NumDealsPurchases', 
              'NumWebPurchases','NumCatalogPurchases', 'NumStorePurchases', 
              'NumWebVisitsMonth','Age', 'Enrollment']
CATEGORICAL_COL = ['Complain', 'Education', 'Marital_Status']
