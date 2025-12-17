# src/preprocessing.py
import numpy as np
import pandas as pd

def _scale_to_minus1_plus1(series: pd.Series) -> pd.Series:
    denom = (series.max() - series.min())
    if denom == 0:
        return series * 0.0  # all same value -> all zeros
    return -1 + 2 * ((series - series.min()) / denom)

def file_to_feature_array(filepath: str, n_points: int = 512) -> np.ndarray:
    """
    Reads one file and returns a feature array of shape (n_points, 3):
      [Field, TopGradient, BottomGradient]

    This is a refactor of your notebook logic.
    """
    df = pd.read_csv(filepath, delimiter="\t")

    # Drop if present (some files might not have it)
    df = df.drop(columns=["Moment [Am^2]"], errors="ignore")

    # Split into top/bottom halves and stitch side-by-side
    half = df.shape[0] // 2
    df1 = df.iloc[:half, :]
    df2 = df.iloc[half:, :].reset_index(drop=True)

    df_combined = pd.concat([df1, df2], axis=1)

    # Reverse first two columns (matches your notebook)
    df_combined.iloc[:, 0] = df_combined.iloc[::-1, 0].values
    df_combined.iloc[:, 1] = df_combined.iloc[::-1, 1].values

    df4 = pd.DataFrame(
        data=df_combined.values,
        columns=["Field (T)", "Fitted Moment (T)", "Field (B)", "Fitted Moment (B)"],
    )

    # Scale each column to [-1, 1]
    for col in df4.columns:
        df4[col] = _scale_to_minus1_plus1(df4[col])

    x_range = np.linspace(-1, 1, n_points)

    top_interp = np.interp(x_range, df4["Field (T)"], df4["Fitted Moment (T)"])
    bot_interp = np.interp(x_range, df4["Field (B)"], df4["Fitted Moment (B)"])

    # Gradients
    top_grad = np.gradient(top_interp, x_range)
    bot_grad = np.gradient(bot_interp, x_range)

    out = np.stack([x_range, top_grad, bot_grad], axis=1).astype(np.float32)
    return out
