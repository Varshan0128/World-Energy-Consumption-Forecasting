from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from src.data_preprocessing import ROOT, OUTPUT_DIR, TARGET, WINDOW, load_selected_series

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
MODELS_DIR = ROOT / "models"

def build_model(kind):
    layers = tf.keras.layers
    inputs = tf.keras.Input(shape=(WINDOW, 1))
    if kind == "SimpleRNN":
        x = layers.SimpleRNN(16, activation="tanh")(inputs)
    elif kind == "LSTM":
        x = layers.LSTM(16)(inputs)
    elif kind == "GRU":
        x = layers.GRU(16)(inputs)
    else:
        raise ValueError(kind)
    x = layers.Dense(8, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mae")
    return model

def make_sequences(values_scaled, years):
    X, y, target_years = [], [], []
    for i in range(WINDOW, len(values_scaled)):
        # Strict annual continuity is ensured in preprocessing.
        X.append(values_scaled[i-WINDOW:i, 0])
        y.append(values_scaled[i, 0])
        target_years.append(int(years[i]))
    return np.asarray(X, dtype=np.float32).reshape(-1, WINDOW, 1), np.asarray(y, dtype=np.float32), np.asarray(target_years)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    series, metadata = load_selected_series()
    years = series["year"].to_numpy()
    values = series[TARGET].to_numpy(dtype=float).reshape(-1, 1)

    n = len(series)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    if train_end <= WINDOW or val_end <= train_end or n - val_end < 3:
        raise ValueError("Not enough annual observations for chronological train/validation/test splits.")

    # Fit scaler only on the training period; never fit on validation/test data.
    scaler = MinMaxScaler()
    scaler.fit(values[:train_end])
    scaled = scaler.transform(values)
    X, y, target_years = make_sequences(scaled, years)

    train_mask = target_years < years[train_end]
    val_mask = (target_years >= years[train_end]) & (target_years < years[val_end])
    test_mask = target_years >= years[val_end]
    if min(train_mask.sum(), val_mask.sum(), test_mask.sum()) == 0:
        raise ValueError("One of the chronological splits has no sequences.")

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    val_years, test_years = target_years[val_mask], target_years[test_mask]

    summary = {
        **metadata,
        "train_target_years": [int(target_years[train_mask].min()), int(target_years[train_mask].max())],
        "validation_target_years": [int(val_years.min()), int(val_years.max())],
        "test_target_years": [int(test_years.min()), int(test_years.max())],
        "scaler_fit_years": [int(years[0]), int(years[train_end-1])],
        "split_method": "chronological target-year split: 70% train, next 15% validation, final 15% test",
        "window": WINDOW,
        "seed": SEED,
    }
    results = []
    histories = {}
    best_kind, best_val_mae = None, float("inf")
    best_model = None

    for kind in ["SimpleRNN", "LSTM", "GRU"]:
        tf.keras.backend.clear_session()
        model = build_model(kind)
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True
        )
        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=250, batch_size=8, shuffle=False, verbose=0,
            callbacks=[callback],
        )
        pred_scaled = model.predict(X_val, verbose=0)
        pred = scaler.inverse_transform(pred_scaled).ravel()
        actual = scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()
        val_mae = float(mean_absolute_error(actual, pred))
        val_rmse = float(np.sqrt(mean_squared_error(actual, pred)))
        results.append({"model": kind, "validation_MAE": val_mae, "validation_RMSE": val_rmse,
                        "epochs_trained": len(history.history["loss"])})
        histories[kind] = history.history
        if val_mae < best_val_mae:
            best_val_mae, best_kind, best_model = val_mae, kind, model

    comparison = pd.DataFrame(results).sort_values("validation_MAE")
    comparison.to_csv(OUTPUT_DIR / "model_comparison_validation.csv", index=False)
    with open(OUTPUT_DIR / "training_histories.json", "w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2)

    # Evaluate selected model once on the untouched test period.
    test_pred_scaled = best_model.predict(X_test, verbose=0)
    test_pred = scaler.inverse_transform(test_pred_scaled).ravel()
    test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    test_mae = float(mean_absolute_error(test_actual, test_pred))
    test_rmse = float(np.sqrt(mean_squared_error(test_actual, test_pred)))

    # Persistence baseline: predict the next value as the final value in the 5-year input window.
    persistence_scaled = X_test[:, -1, 0].reshape(-1, 1)
    persistence_pred = scaler.inverse_transform(persistence_scaled).ravel()
    baseline_mae = float(mean_absolute_error(test_actual, persistence_pred))
    baseline_rmse = float(np.sqrt(mean_squared_error(test_actual, persistence_pred)))

    pred_df = pd.DataFrame({
        "year": target_years[test_mask],
        "actual": test_actual,
        "predicted": test_pred,
        "absolute_error": np.abs(test_actual - test_pred),
        "persistence_baseline": persistence_pred,
        "baseline_absolute_error": np.abs(test_actual - persistence_pred),
    })
    pred_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    best_model.save(MODELS_DIR / "best_model.keras")
    joblib.dump(scaler, MODELS_DIR / "target_scaler.joblib")
    final_report = {
        **summary,
        "selected_model_by_validation_MAE": best_kind,
        "test_MAE": test_mae,
        "test_RMSE": test_rmse,
        "persistence_test_MAE": baseline_mae,
        "persistence_test_RMSE": baseline_rmse,
        "target_units": "same units as dataset target; verify dataset metadata",
        "test_observations": int(len(test_actual)),
        "note": "Metrics are measured on the untouched chronological test period. Small annual datasets make results uncertain."
    }
    (OUTPUT_DIR / "evaluation_metrics.json").write_text(json.dumps(final_report, indent=2), encoding="utf-8")

    plt.figure(figsize=(10, 5))
    plt.plot(pred_df["year"], pred_df["actual"], marker="o", label="Actual")
    plt.plot(pred_df["year"], pred_df["predicted"], marker="o", label=f"Predicted ({best_kind})")
    plt.plot(pred_df["year"], pred_df["persistence_baseline"], linestyle="--", label="Persistence baseline")
    plt.xlabel("Year")
    plt.ylabel(TARGET)
    plt.title(f"{summary['selected_country']}: held-out energy consumption forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    for kind, hist in histories.items():
        plt.plot(hist["loss"], label=f"{kind} train")
        plt.plot(hist["val_loss"], linestyle="--", label=f"{kind} validation")
    plt.xlabel("Epoch")
    plt.ylabel("MAE loss (scaled target)")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=160)
    plt.close()

    print(json.dumps(final_report, indent=2))
    print("\nValidation comparison:\n", comparison.to_string(index=False))
    print(f"\nSaved model: {MODELS_DIR / 'best_model.keras'}")

if __name__ == "__main__":
    main()
