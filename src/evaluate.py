from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from config import config

def get_metrics(y_pred, y_test, classes=config.CLASSES, average=None):
    """Performance metrics using ground truths and predictions."""
    # Performance
    metrics = {"overall": {}, "class": {}}

    # Overall metrics
    overall_metrics = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    metrics["overall"]["precision"] = overall_metrics[0]
    metrics["overall"]["recall"] = overall_metrics[1]
    metrics["overall"]["f1"] = overall_metrics[2]
    metrics["overall"]["num_samples"] = np.float64(len(y_test))

    # Per-class metrics
    class_metrics = precision_recall_fscore_support(y_test, y_pred, average=None)
    for i, _class in enumerate(classes):
        metrics["class"][_class] = {
            "precision": class_metrics[0][i],
            "recall": class_metrics[1][i],
            "f1": class_metrics[2][i],
            "num_samples": np.float64(class_metrics[3][i]),
        }

    return metrics