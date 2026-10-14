import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from src.data_preprocessing import ROOT, OUTPUT_DIR, TARGET, WINDOW

def main():
    parser = argparse.ArgumentParser(description="Recursive multi-year energy forecast.")
    parser.add_argument("--years", type=int, default=1)
    args = parser.parse_args()
    if args.years < 1:
        raise ValueError("--years must be >= 1")
    series_path = OUTPUT_DIR / "selected_country_series.csv"
    if not series_path.exists():
        raise FileNotFoundError("Run `python -m src.train` first.")
    series = pd.read_csv(series_path)
    values = series[TARGET].to_numpy(dtype=float).reshape(-1, 1)
    model = tf.keras.models.load_model(ROOT / "models" / "best_model.keras")
    scaler = joblib.load(ROOT / "models" / "target_scaler.joblib")
    window = scaler.transform(values[-WINDOW:]).reshape(1, WINDOW, 1)
    future = []
    next_year = int(series["year"].max()) + 1
    for step in range(args.years):
        pred_scaled = model.predict(window, verbose=0)
        pred = float(scaler.inverse_transform(pred_scaled)[0, 0])
        future.append({"year": next_year + step, "predicted_" + TARGET: pred})
        window = np.concatenate([window[:, 1:, :], pred_scaled.reshape(1, 1, 1)], axis=1)
    out = pd.DataFrame(future)
    out.to_csv(OUTPUT_DIR / "future_forecast.csv", index=False)
    print(out.to_string(index=False))
    print("\nNote: forecasts beyond one year are recursive and uncertainty compounds.")
if __name__ == "__main__":
    main()
