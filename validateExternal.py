import random
import os
import numpy as np
import pandas as p
from syntheticPoint import get_delta_h, get_delta_d, find_altitude_angle, TELESCOPE_LOC, ARCSEC_TO_RAD, PHI
from syntheticPoint import IH, ID, CH, NP, MA, ME, TF, PHH, PDD

from astropy.io import fits
from astropy.coordinates import SkyCoord, CIRS
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u

from datetime import datetime, timedelta
from astropy.time import Time

#import requests
import xgboost as xgb

###Workflow
# Randomly generate roll/pitch values (can copy over logic from syntheticPoint.py)
# Calculate true offsets (get_delta functions, already written)
# Create feature vector of inputs that predict.py takes in to run model and output predictions
    #inputs: year, month, day, hour, minute, second, lst hours, obs_ra_deg, obs_dec_deg, previous_acq_error_ra, previous_acq_error_dec
# Call API, input feature vector
# Receive predicted offsets, accumulate true correction vs predicted correction and calculate residuals
# Plot residuals (done externally, not in script)
# ###


def main():

    ###start by randomly generating all the data you need
    start_time = datetime(2026, 6, 24, 21, 0, 0)
    current_time = start_time
    dataset = [] #dictionary
    #changing from for loop to while loop to account for data that might be dropped because of restrictions
    model_ra = xgb.XGBRegressor()
    model_ra.load_model(os.path.expanduser("~/Documents/Portfolio/telescope-api-main/api/models/model_ra.json"))

    model_dec = xgb.XGBRegressor()
    model_dec.load_model(os.path.expanduser("~/Documents/Portfolio/telescope-api-main/api/models/model_dec.json"))

    while len(dataset) < 1000:
        #lst_float = np.random.uniform(0.0, 24.0)
        #generating a random lst that does not sync with the observation time will corrupt data

        icrs_ra_float = np.random.uniform(0.0, 24.0)
        icrs_dec_float = np.random.uniform(-65.0, 80.0)

        icrs_coord = SkyCoord(ra=icrs_ra_float * u.hourangle, dec=icrs_dec_float * u.deg, frame='icrs')

        #need an observation timestamp, incremented at regular intervals, BUT it's better to have randomized timestamps i think
        #SO let's add some random sub-minute noise
        #obs_time = Time(start_time + timedelta(minutes=3 * i)) --scratch this

        #originally we were incrementing by 3 minutes: let's randomize those minutes as well, making sure they increment chronogically
        #because the model is autoregressive, so it trains based on PREVIOUS predictions

        #but we can't just randomize minutes themselves, so lets randomize the gap between minutes (timedelta)

        minute_gap = random.randint(1, 10)
        random_seconds = random.randint(0, 59)

        current_time += timedelta(minutes = minute_gap, seconds = random_seconds)
        obs_time = Time(current_time)

        cirs_coord = icrs_coord.transform_to(CIRS(obstime = obs_time, location = TELESCOPE_LOC))

        target_ra_apparent = cirs_coord.ra.hour
        target_dec_apparent = cirs_coord.dec.deg

        lst_float = obs_time.sidereal_time('apparent', longitude=TELESCOPE_LOC.lon).hour

        pitch = target_dec_apparent
        roll = lst_float - target_ra_apparent

        if roll > 12.0:
            roll -= 24.0
        elif roll < -12.0:
            roll += 24.0

        #dropping any stars that are below Hawaii's horizon
        if roll < -5.0 or roll > 5.0:
            continue

        pitch_rad = np.radians(pitch)
        roll_deg = roll * 15.0
        roll_rad = np.radians(roll_deg)

        horizon_filter = find_altitude_angle(pitch_rad, roll_rad)

        #for Haleakala, the horizon limit is 15 degrees
        if horizon_filter < 15.0:
            continue


        ### Calculating true offsets
        total_offset_h = get_delta_h(pitch_rad, roll_rad)
        total_offset_d = get_delta_d(pitch_rad, roll_rad)

        offset_h_hours = np.degrees(total_offset_h) / 15.0
        offset_d_degrees = np.degrees(total_offset_d)

        #don't need the actual coordinates for this one i think
        # actual_apparent_ra = target_ra_apparent + offset_h_hours
        # actual_apparent_dec = target_dec_apparent + offset_d_degrees

        #extract the parts of the observation time/date

        obs_year = current_time.year
        obs_month = current_time.month
        obs_day = current_time.day
        obs_hour = current_time.hour
        obs_minute = current_time.minute
        obs_second = current_time.second

        ###Create feature vector (list)
         #inputs: year, month, day, hour, minute, second, lst hours, obs_ra_deg, obs_dec_deg, previous_acq_error_ra, previous_acq_error_dec
        row = {
            #feature vector
            "year": obs_year,
            "month": obs_month,
            "day": obs_day,
            "hour": obs_hour,
            "minute": obs_minute,
            "second": obs_second,
            "lst_hours": lst_float,
            "obs_ra_deg": target_ra_apparent * 15.0,
            "obs_dec_deg": target_dec_apparent,
            "lat_deg": 20.7081,
            "lon_deg": -156.2567,
            "elevation_m": 3048.0,
            
            #don't need to keep track of previous_acq_error variables, those are taken care of internally within Docker
            #prev_vals = read_previous_from_log(args.log_file) ---internal stuff that is invisible to me

            #ground truth
            "total_offset_roll": offset_h_hours * 15.0, #in degrees
            "total_offset_pitch": offset_d_degrees #in degrees
        }

        # response = requests.post("http://localhost:5001/predict", json=row)
        # result = response.json()

        # predicted_ra_offset_deg = result["prediction"]["offsets_equatorial"]["ra_deg"]
        # predicted_dec_offset_deg = result["prediction"]["offsets_equatorial"]["dec_deg"]


        X = np.array([[lst_float, target_ra_apparent * 15.0, target_dec_apparent,
        np.sin(roll_rad), np.cos(roll_rad), np.sin(pitch_rad), np.cos(pitch_rad),
        0.0, 0.0]])# previous_acq_error initialized to 0

        predicted_ra_offset_deg = float(model_ra.predict(X)[0])
        predicted_dec_offset_deg = float(model_dec.predict(X)[0])

        #true offset value needs to be negated
        #In syntheticPoint.py: actual_apparent_dec = target_dec_apparent + offset_d_degrees
        #So solv_dec = obs_dec + offset_d
        #but in booster.py: y = obs_dec - solv_dec
        #y = obs_dec - (obs_dec + offset_d) = -offset_d
        #So the model learned to predict negative offset_d. 
        # But in the validation script you're comparing against +offset_d_degrees as the ground truth. That's the sign mismatch.

        residual_ra_deg = (offset_h_hours*15.0) - predicted_ra_offset_deg
        residual_dec_deg = offset_d_degrees - predicted_dec_offset_deg

        row.update({
            "predicted_roll_offset": predicted_ra_offset_deg,
            "predicted_pitch_offset": predicted_dec_offset_deg,
            "residual_ra": residual_ra_deg,
            "residual_dec": residual_dec_deg,
            "roll_hours": roll,
            "pitch_deg": pitch,
            "ch": CH * (1.0 / np.cos(pitch_rad)),
            "ih": +IH,
            "mah": -MA * np.cos(roll_rad) * np.tan(pitch_rad),
            "meh": ME * np.sin(roll_rad) * np.tan(pitch_rad),
            "nP": NP * np.tan(pitch_rad),
            "tfh": TF * np.cos(PHI) * np.sin(roll_rad) * (1/np.cos(pitch_rad)),
            "phh": PHH * roll_rad,
            "id": ID,
            "mad": MA * np.sin(roll_rad),
            "med": ME * np.cos(roll_rad),
            "tfd": TF * ((np.cos(PHI)*np.cos(roll_rad)*np.sin(pitch_rad)) - (np.sin(PHI)*np.cos(pitch_rad))),
            "pdd": PDD * pitch_rad
        })

        dataset.append(row)

        if len(dataset) % 500 == 0:
            print(f"{len(dataset)} / 1000 complete")

    data = p.DataFrame(dataset)
    columns = [
            "year", "month", "day", "hour", "minute", "second", "lst_hours", 
            "obs_ra_deg", "obs_dec_deg", "lat_deg", "lon_deg", "elevation_m",
            "total_offset_roll", "total_offset_pitch", "predicted_roll_offset", "predicted_pitch_offset", "residual_ra", "residual_dec", "roll_hours", "pitch_deg",
            "ch", "ih", "mah", "meh", "nP", "tfh", "phh",
            "id", "mad", "med", "tfd", "pdd"
        ]
    data = data[columns]
    print("\n Results")
    print(data.head())

    data.to_csv(os.path.expanduser("~/Documents/synthetic-pointer-data/residuals_data.csv"), sep=" ", index=False)

    print("\nsaved to csv")



if __name__ == "__main__":
    main()
