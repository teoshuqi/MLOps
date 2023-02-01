from typing import List

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from config import config


def get_metrics(y_pred: np.ndarray, y_test: np.ndarray, classes: List = config.CLASSES):
    """
    Performance metrics using ground truths and predictions.

    Args:
        y_true (np.ndarray): true labels.
        y_pred (np.ndarray): predicted labels.
        classes (List): list of class labels.
    Returns:
        Dict: performance metrics.
    """
    y_test_str = [config.CLASSES[i] for i in y_test]
    # Performance
    metrics = {"overall": {}, "class": {}}

    # Overall metrics
    overall_metrics = precision_recall_fscore_support(y_test_str, y_pred, average="weighted")
    metrics["overall"]["precision"] = overall_metrics[0]
    metrics["overall"]["recall"] = overall_metrics[1]
    metrics["overall"]["f1"] = overall_metrics[2]
    metrics["overall"]["num_samples"] = np.float64(len(y_test_str))

    # Per-class metrics
    class_metrics = precision_recall_fscore_support(y_test_str, y_pred, average=None)
    for i, _class in enumerate(classes):
        metrics["class"][_class] = {
            "precision": class_metrics[0][i],
            "recall": class_metrics[1][i],
            "f1": class_metrics[2][i],
            "num_samples": np.float64(class_metrics[3][i]),
        }

    return metrics
