from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
TARGET = "primary_energy_consumption"
MIN_YEARS = 25
WINDOW = 5

def find_csv():
    files = sorted(p for p in RAW_DIR.glob("*.csv") if p.is_file())
    if not files:
        raise FileNotFoundError(
            f"No CSV found in {RAW_DIR}. Download the Kaggle dataset and place its CSV there."
        )
    # Prefer a file with the expected target column.
    for path in files:
        try:
            cols = pd.read_csv(path, nrows=0).columns
            if TARGET in cols:
                return path
        except Exception:
            continue
    raise ValueError(f"No CSV in {RAW_DIR} contains required column {TARGET!r}.")

def standardize_columns(df):
    aliases = {
        "country": ["country", "entity", "location", "country_name"],
        "year": ["year", "year_num"],
    }
    lower = {str(c).strip().lower(): c for c in df.columns}
    rename = {}
    for canonical, choices in aliases.items():
        for choice in choices:
            if choice in lower:
                rename[lower[choice]] = canonical
                break
    df = df.rename(columns=rename)
    required = {"country", "year", TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}. Available columns: {list(df.columns)}")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=["country", "year"])
    df["year"] = df["year"].astype(int)
    return df

def longest_consecutive_segment(group):
    group = group.dropna(subset=[TARGET]).sort_values("year")
    group = group.drop_duplicates("year", keep="last")
    if group.empty:
        return group
    years = group["year"].to_numpy()
    breaks = np.where(np.diff(years) != 1)[0] + 1
    segments = np.split(group, breaks)
    return max(segments, key=len)

def load_selected_series():
    path = find_csv()
    df = pd.read_csv(path)
    df = standardize_columns(df)
    candidates = []
    for country, group in df.groupby("country", sort=True):
        segment = longest_consecutive_segment(group)
        if len(segment) >= MIN_YEARS:
            candidates.append((len(segment), str(country), segment))
    if not candidates:
        raise ValueError(
            f"No country has at least {MIN_YEARS} consecutive years with a non-missing {TARGET}. "
            "Inspect the dataset and lower MIN_YEARS only with a documented rationale."
        )
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, country, selected = candidates[0]
    selected = selected[["year", TARGET]].sort_values("year").reset_index(drop=True)
    if selected["year"].duplicated().any() or not np.all(np.diff(selected["year"]) == 1):
        raise AssertionError("Selected series must have unique, consecutive annual records.")
    metadata = {
        "source_file": path.name,
        "selected_country": country,
        "first_year": int(selected.year.min()),
        "last_year": int(selected.year.max()),
        "consecutive_years": int(len(selected)),
        "target": TARGET,
        "selection_rule": f"Longest consecutive non-missing target segment; minimum {MIN_YEARS} years",
        "input_window_years": WINDOW,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUTPUT_DIR / "selected_country_series.csv", index=False)
    (OUTPUT_DIR / "dataset_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return selected, metadata
