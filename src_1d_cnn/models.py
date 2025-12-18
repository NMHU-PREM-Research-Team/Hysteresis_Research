# src_1d_cnn/models.py
from keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_cnn1d_model(timesteps: int, n_features: int, n_classes: int, lr: float = 1e-3):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation="relu", input_shape=(timesteps, n_features)),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        Conv1D(32, kernel_size=3, activation="relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        Flatten(),
        Dense(50, activation="relu"),
        Dropout(0.2),
        Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
