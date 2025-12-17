# src/metrics.py
import numpy as np
import pandas as pd

def build_metrics_df(accuracy_scores, f1_scores, recall_scores, precision_scores) -> pd.DataFrame:
    num_folds = len(accuracy_scores)
    df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(num_folds)],
        "Accuracy": accuracy_scores,
        "F1-score": f1_scores,
        "Recall": recall_scores,
        "Precision": precision_scores,
    })

    avg_row = {
        "Fold": "Average",
        "Accuracy": float(np.mean(accuracy_scores)),
        "F1-score": float(np.mean(f1_scores)),
        "Recall": float(np.mean(recall_scores)),
        "Precision": float(np.mean(precision_scores)),
    }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    df[["Accuracy", "F1-score", "Recall", "Precision"]] = df[["Accuracy", "F1-score", "Recall", "Precision"]].round(4)
    return df
