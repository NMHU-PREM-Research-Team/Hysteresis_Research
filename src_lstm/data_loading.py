# src_lstm/data_loading.py
import os
import numpy as np
import tensorflow as tf

from .preprocessing import file_to_feature_array

def load_class_dir(data_root: str, class_dir: str, label: int, n_points: int):
    X_list, y_list = [], []
    folder = os.path.join(data_root, class_dir)

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Missing class folder: {folder}")

    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue

        X_list.append(file_to_feature_array(fpath, n_points=n_points))
        y_list.append(label)

    return X_list, y_list

def build_X_y(data_root: str, class_map: dict[str, int], n_points: int, n_classes: int):
    """
    class_map example:
      {"waspwaisted": 0, "gooseneck": 1, "regular": 2}
    """
    X_all, y_all = [], []

    for class_dir, label in class_map.items():
        X_list, y_list = load_class_dir(data_root, class_dir, label, n_points)
        X_all.extend(X_list)
        y_all.extend(y_list)

    X = np.asarray(X_all, dtype=np.float32)              
    y = np.asarray(y_all, dtype=int).reshape(-1, 1)      
    y_cat = tf.keras.utils.to_categorical(y, num_classes=n_classes)

    return X, y, y_cat
