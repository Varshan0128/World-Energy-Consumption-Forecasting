from pathlib import Path
import json
import pandas as pd
from src.data_preprocessing import ROOT, OUTPUT_DIR

def main():
    metrics_path = OUTPUT_DIR / "evaluation_metrics.json"
    predictions_path = OUTPUT_DIR / "test_predictions.csv"
    if not metrics_path.exists() or not predictions_path.exists():
        raise FileNotFoundError("Run `python -m src.train` first.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    preds = pd.read_csv(predictions_path)
    print("Evaluation metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nHeld-out predictions:")
    print(preds.to_string(index=False))

if __name__ == "__main__":
    main()
