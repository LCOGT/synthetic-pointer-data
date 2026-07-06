import xgboost as xgb
import numpy as np

model_ra = xgb.XGBRegressor()
model_ra.load_model("models/model_ra.json")

model_dec = xgb.XGBRegressor()
model_dec.load_model("models/model_dec.json")

feature_names = [
    "lst_hours", "obs_ra_deg", "obs_dec_deg",
    "sin_roll", "cos_roll", "sin_pitch", "cos_pitch",
    "tan_pitch", "sec_pitch",
    "previous_acq_error_ra", "previous_acq_error_dec"
]

print("=== RA Model Feature Importance ===")
for name, score in sorted(zip(feature_names, model_ra.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:>24s}: {score:.4f}")

print()
print("=== Dec Model Feature Importance ===")
for name, score in sorted(zip(feature_names, model_dec.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:>24s}: {score:.4f}")