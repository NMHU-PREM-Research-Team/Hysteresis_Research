# src_lstm/viz.py
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_loss_plot(history, fold: int, outpath: str):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss for Fold {fold}")
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01)
    plt.close()

def save_accuracy_plot(history, fold: int, outpath: str):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"Accuracy for Fold {fold}")
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01)
    plt.close()

def save_cm_plot(conf_matrix, labels, fold: int, outpath: str):
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for Fold {fold}")
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01)
    plt.close()

def save_metrics_table_as_svg(df, filename: str):
    n_rows, n_cols = df.shape
    fig, ax = plt.subplots(figsize=(n_cols * 2.0, n_rows * 0.5 + 1))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.2)

    plt.tight_layout()
    plt.savefig(filename, format="svg", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
