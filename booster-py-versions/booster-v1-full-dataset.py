#!/usr/bin/env python3
"""
booster.py — VERSION 1 SNAPSHOT
Trains on the full 20k-row dataset with no held-out split.
Original 11-feature schema (year, month, day, hour, minute, second,
lst_hours, obs_ra_deg, obs_dec_deg, previous_acq_error_ra, previous_acq_error_dec).
Superseded by the restored-split version and later the trig-feature version.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import h5py
import xgboost as xgb

Path("plots").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("plots") / ".matplotlib"))
import matplotlib.pyplot as plt


DATA_FILE = "data/data.h5"
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")

FEATURE_NAMES = (
    "year", "month", "day",
    "hour", "minute", "second",
    "lst_hours",
    "obs_ra_deg", "obs_dec_deg",
    "previous_acq_error_ra", "previous_acq_error_dec",
)


def load_data(h5_file_path):
    """Load array from HDF5 file under dataset 'data'."""
    with h5py.File(h5_file_path, 'r') as hf:
        return hf['data'][:]


def sort_by_datetime(data):
    """
    Lexicographically sort rows by the first six columns
    [year, month, day, hour, minute, second].
    """
    years, months, days = data[:,0], data[:,1], data[:,2]
    hours, mins, secs   = data[:,3], data[:,4], data[:,5]
    idx = np.lexsort((secs, mins, hours, days, months, years))
    return data[idx]


def make_features_and_labels(split_data):
    """
    Build model inputs and labels in the same order used by predict.py.

    Inputs X:
      year, month, day, hour, minute, second,
      lst_hours, obs_ra_deg, obs_dec_deg,
      previous_acq_error_ra, previous_acq_error_dec

    Labels y: shape (M,2) = [obs_RA - solv_RA, obs_DEC - solv_DEC].

    Column indices:
       7:obs_RA  8:obs_DEC  9:solv_RA  10:solv_DEC
    """
    base_features = split_data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]

    obs_ra, obs_dec = split_data[:, 7], split_data[:, 8]
    solv_ra, solv_dec = split_data[:, 9], split_data[:, 10]

    current_error_ra = obs_ra - solv_ra
    current_error_dec = obs_dec - solv_dec

    previous_error_ra = np.roll(current_error_ra, 1)
    previous_error_dec = np.roll(current_error_dec, 1)
    previous_error_ra[0] = 0.0
    previous_error_dec[0] = 0.0

    X = np.column_stack([base_features, previous_error_ra, previous_error_dec])
    y = np.column_stack([obs_ra - solv_ra, obs_dec - solv_dec])
    return X, y


def parse_args():
    parser = argparse.ArgumentParser(description="Train autoregressive XGBoost pointing-offset models (full dataset, no split).")
    parser.add_argument("--data-file", default=DATA_FILE, help=f"HDF5 file from load_out.py (default: {DATA_FILE})")
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_data(args.data_file)
    data = sort_by_datetime(data)

    X, y = make_features_and_labels(data)
    print(f"Full dataset: X={X.shape}, y={y.shape}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_ra = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=3,
        objective='reg:absoluteerror',
        verbosity=0,
    )
    model_ra.fit(X, y[:, 0], verbose=False)
    model_ra.save_model(str(MODELS_DIR / "model_ra.json"))
    print("model_ra.json saved.")

    model_dec = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=3,
        objective='reg:absoluteerror',
        verbosity=0,
    )
    model_dec.fit(X, y[:, 1], verbose=False)
    model_dec.save_model(str(MODELS_DIR / "model_dec.json"))
    print("model_dec.json saved.")


if __name__ == "__main__":
    main()