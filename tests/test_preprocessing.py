import numpy as np
from src.train import make_sequences
from src.data_preprocessing import WINDOW

def test_sequence_length_and_next_year_targets():
    values = np.arange(12, dtype=float).reshape(-1, 1)
    years = np.arange(2000, 2012)
    X, y, target_years = make_sequences(values, years)
    assert X.shape == (12 - WINDOW, WINDOW, 1)
    assert y.shape == (12 - WINDOW,)
    assert target_years[0] == 2005
    assert np.allclose(X[0, :, 0], [0, 1, 2, 3, 4])
    assert y[0] == 5

def test_years_are_chronological():
    values = np.arange(10, dtype=float).reshape(-1, 1)
    years = np.arange(1990, 2000)
    _, _, target_years = make_sequences(values, years)
    assert np.all(np.diff(target_years) == 1)
