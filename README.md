# World Energy Consumption Forecasting

A reproducible TensorFlow/Keras time-series project comparing SimpleRNN, LSTM, and GRU models for one-year-ahead country-level primary energy consumption forecasting.

## Problem statement
Use the previous five consecutive years to predict the next year's `primary_energy_consumption`. Sort observations chronologically, handle missing data, normalize using training data only, and evaluate with MAE and RMSE on an untouched chronological test period.

## 1. Environment (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Download the Kaggle dataset

Dataset: https://www.kaggle.com/datasets/patricklford/world-energy-consumption

Download the CSV from Kaggle and place it in `data/raw/`. Do not rename columns. You may also use Kaggle CLI if configured:

```powershell
pip install kaggle
kaggle datasets download -d patricklford/world-energy-consumption -p data/raw --unzip
```

Kaggle CLI requires your own Kaggle API credentials. Never commit `kaggle.json` or any API token.

## 3. Run the pipeline

```powershell
python -m src.train
python -m src.evaluate
python -m src.predict --years 3
python -m pytest
```

`src.train` finds a CSV in `data/raw`, identifies common country/year column names, and requires `primary_energy_consumption`. It selects the country with the longest consecutive run of valid target observations (minimum 25 years), then uses the last 15% of target years for test and the preceding 15% for validation. The remaining earlier target years are used for training.

Outputs are written to `outputs/`; trained models and the scaler are written to `models/`. Actual model metrics are generated only after running the pipeline on the real downloaded dataset.

## Evaluation design

- One feature: historical `primary_energy_consumption` (univariate forecast).
- Input window: five consecutive annual observations.
- Forecast horizon: one year.
- Scaling: `MinMaxScaler` fit on training-period observations only.
- Model selection: lowest validation MAE among SimpleRNN, LSTM, and GRU.
- Final evaluation: held-out chronological test period, reported in original target units.
- Baseline: persistence forecast (next year equals the most recent observed year).
- Test windows may use the previous observed actual values, including values from the end of the training/validation period. This is a rolling one-step-ahead evaluation and does not expose future targets to the model.
- The test set is not used to select the architecture.

## Important limitations

Annual country-level data is small. RNN/LSTM/GRU performance can be unstable and may not outperform the persistence baseline. Do not claim that a model is accurate until its held-out metrics are measured. MAE/RMSE are in the dataset's target units (commonly TWh for OWID-derived data; verify the exact dataset/version and metadata).

## GitHub

After verifying the project locally:

```powershell
git init
git branch -M main
git add README.md requirements.txt .gitignore src tests
git status
git commit -m "Add world energy forecasting pipeline"
git switch -c feature/world-energy-forecasting
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin feature/world-energy-forecasting
```

Create an empty GitHub repository first. Replace the remote URL with your own. If this folder is already inside an existing repository, do not run `git init` or change branches until you inspect `git status` and `git remote -v`.
