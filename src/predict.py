import numpy as np
import pandas as pd
from src import data
from config import config
from typing import Dict, List

def custom_predict(y_prob: np.ndarray, threshold: float) -> List:
    """
    Custom predict function that defaults to an index if conditions are not met.

    Args:
        y_prob (np.ndarray): predicted probabilities
        threshold (float): minimum softmax score to predict majority class
        index (int): label index to use if custom conditions is not met.
    
    Returns:
        list : predicted class labels.
    """
    y_pred = [ config.CLASSES[p[0] > threshold] for p in y_prob]
    return y_pred

def predict(customer_data: pd.DataFrame, artifacts: Dict) -> List:
    """
    Predict customer acceptance of new superstore marketing campaign.

    Args:
        customer_data (List): raw customer data to classify
        artifcats (Dict)L artificats from a run
    
    Returns:
        List: predictions for input customer data
    """
    x = data.preprocess_predict(customer_data, artifacts)
    selected_columns = ['NumCatalogPurchases', 'NumStorePurchases','MntWines', 'MntMeatProducts','Recency',
                      'Age', 'Income','NumWebVisitsMonth','Education', 'MntGoldProds', 'MntSweetProducts']
    x_selected = x.loc[:, selected_columns]
    y_pred = custom_predict(
        y_prob=artifacts["model"].predict_proba(x_selected),
        threshold=artifacts["args"].threshold)
    predictions = [
        {
            "input_text": customer_data.iloc[i,:].to_dict(),
            "predicted_tags": y_pred[i],
        }
        for i in range(len(y_pred))
    ]
    config.logger.info(predictions)
    return predictions
