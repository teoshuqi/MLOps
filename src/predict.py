import numpy as np
from src import data
from config import config

def custom_predict(y_prob, threshold):
    """Custom predict function that defaults
    to an index if conditions are not met."""
    y_pred = [ config.CLASSES[p[0] > threshold] for p in y_prob]
    return y_pred

def predict(customer_data, artifacts):
    """Predict customer acceptance of new superstore marketing campaign """
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
