print(">>> train_kfold.py started")
# src_lstm/train_kfold.py
import os
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .config import Config
from .data_loading import build_X_y
from .models import create_lstm_model
from .viz import (
    ensure_dir,
    save_loss_plot,
    save_accuracy_plot,
    save_cm_plot,
    save_metrics_table_as_svg,
)
from .metrics import build_metrics_df

def main():
    cfg = Config()
    ensure_dir(cfg.results_dir)

    class_map = {
        cfg.wasp_dir: 0,
        cfg.goose_dir: 1,
        cfg.regular_dir: 2,
    }

    X, y_int, y_cat = build_X_y(
        data_root=cfg.data_root,
        class_map=class_map,
        n_points=cfg.n_points,
        n_classes=cfg.n_classes,
    )

    timesteps, n_features = X.shape[1], X.shape[2]
    labels = ["Wasp-Waisted", "Gooseneck", "Regular"]

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)

    accuracy_scores, f1_scores, recall_scores, precision_scores = [], [], [], []

    fold = 1
    for train_idx, test_idx in skf.split(X, y_int):
        print(f"\nFold {fold}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_cat[train_idx], y_cat[test_idx]

        model = create_lstm_model(timesteps, n_features, cfg.n_classes, lr=cfg.lr)

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=cfg.es_patience,
            restore_best_weights=True,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.rlrop_factor,
            patience=cfg.rlrop_patience,
            min_lr=cfg.min_lr,
        )

        history = model.fit(
            X_train,
            y_train,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1,
        )

        # Save fold curves
        save_accuracy_plot(history, fold, outpath=os.path.join(cfg.results_dir, f"{cfg.model_name}_acc_fold_{fold}.svg"))
        save_loss_plot(history, fold, outpath=os.path.join(cfg.results_dir, f"{cfg.model_name}_loss_fold_{fold}.svg"))

        # Predictions
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        save_cm_plot(cm, labels, fold, outpath=os.path.join(cfg.results_dir, f"{cfg.model_name}_CM_fold_{fold}.svg"))

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        rec = recall_score(y_true, y_pred, average="weighted")
        prec = precision_score(y_true, y_pred, average="weighted")

        print(f"Fold {fold} Accuracy:  {acc:.4f}")
        print(f"Fold {fold} F1-score:  {f1:.4f}")
        print(f"Fold {fold} Recall:    {rec:.4f}")
        print(f"Fold {fold} Precision: {prec:.4f}")

        accuracy_scores.append(acc)
        f1_scores.append(f1)
        recall_scores.append(rec)
        precision_scores.append(prec)

        fold += 1

    print("\nAverages across folds")
    print(f"Average Accuracy:  {np.mean(accuracy_scores):.4f}")
    print(f"Average F1-score:  {np.mean(f1_scores):.4f}")
    print(f"Average Recall:    {np.mean(recall_scores):.4f}")
    print(f"Average Precision: {np.mean(precision_scores):.4f}")

    metrics_df = build_metrics_df(accuracy_scores, f1_scores, recall_scores, precision_scores)
    metrics_csv = os.path.join(cfg.results_dir, "lstm_kfold_metrics.csv")
    metrics_svg = os.path.join(cfg.results_dir, "lstm_kfold_metrics.svg")

    metrics_df.to_csv(metrics_csv, index=False)
    save_metrics_table_as_svg(metrics_df, filename=metrics_svg)

    print(f"\nSaved: {metrics_csv}")
    print(f"Saved: {metrics_svg}")

if __name__ == "__main__":
    main()
