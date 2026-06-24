#!/usr/bin/env python3
import os
import argparse
import random
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.coordinates import SkyCoord, CIRS
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u

from datetime import datetime, timedelta
from astropy.time import Time

### Workflow 
# Pick an LST (randomly generated)
# Generate ICRS coordinates
# ICRS -> Apparent
# Apparent -> roll and pitch (roll = LST - Target RA, pitch = DEC) -- equatorial mount of telescope
# from roll and pitch -> input into Tpoint formulas, get RA and Dec offsets (∆h, ∆∂)
# compute actual RA and DEC (target + offset)
# convert back to apparent coordinates (where telescope is actually pointing)
# Table: Roll, Pitch, ∆h, ∆∂, LST
# Table: Target RA, Target DEC (star), Actual RA, Actual DEC (telescope), LST
# ###

#telescope location: Haleakala Site in Maui, Hawaii: ogg.clma.2m0a
TELESCOPE_LOC = EarthLocation(lon=-156.2567 * u.deg, lat=20.7081 * u.deg, height=3048 * u.m)
#define Tpoint constants
ARCSEC_TO_RAD = np.radians(1.0 / 3600.0)
PHI = np.radians(20.7081) #specific to the telescope located at the Haleakala site in Maui, Hawaii

IH= 15.0 * ARCSEC_TO_RAD
ID = -20.0 * ARCSEC_TO_RAD
CH = 10.0 * ARCSEC_TO_RAD
NP = -8.0 * ARCSEC_TO_RAD
MA = 35.0 * ARCSEC_TO_RAD
ME = -45.0 * ARCSEC_TO_RAD
TF = 25.0 * ARCSEC_TO_RAD
PHH = 5.0 * ARCSEC_TO_RAD
PDD = -4.0 * ARCSEC_TO_RAD

def get_delta_h(pitch, roll):
    ch = CH * (1.0 / np.cos(pitch))
    ih = +IH
    mah = -MA * np.cos(roll) * np.tan(pitch)
    meh = ME * np.sin(roll) * np.tan(pitch)
    nP = NP * np.tan(pitch)
    tfh = TF * np.cos(PHI) * np.sin(roll) * (1/np.cos(pitch))
    phh = PHH * roll
    return ch + ih + mah + meh + nP + tfh + phh

def get_delta_d(pitch, roll):
    id = ID
    mad = MA * np.sin(roll)
    med = ME * np.cos(roll)
    tfd = TF * ((np.cos(PHI)*np.cos(roll)*np.sin(pitch)) - (np.sin(PHI)*np.cos(pitch)))
    pdd = PDD * pitch
    return id + mad + med + tfd + pdd

    
def main():
    start_time = datetime(2026, 6, 24, 21, 0, 0)
    dataset = []
    for i in range(2000):
        lst_float = np.random.uniform(0.0, 24.0)

        icrs_ra_float = np.random.uniform(0.0, 24.0)
        icrs_dec_float = np.random.uniform(-65.0, 80.0)

        icrs_coord = SkyCoord(ra=icrs_ra_float * u.hourangle, dec=icrs_dec_float * u.deg, frame='icrs')

        #need an observation timestamp
        obs_time = Time(start_time + timedelta(minutes=3 * i))

        cirs_coord = icrs_coord.transform_to(CIRS(obstime = obs_time, location = TELESCOPE_LOC))

        target_ra_apparent = cirs_coord.ra.hour
        target_dec_apparent = cirs_coord.dec.deg

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


        total_offset_h = get_delta_h(pitch_rad, roll_rad)
        total_offset_d = get_delta_d(pitch_rad, roll_rad)

        offset_h_hours = np.degrees(total_offset_h) / 15.0
        offset_d_degrees = np.degrees(total_offset_d)

        actual_apparent_ra = target_ra_apparent + offset_h_hours
        actual_apparent_dec = target_dec_apparent + offset_d_degrees

        #add a row of data to the dictionary
        dataset.append({
            "LST": lst_float,
            "Roll": roll,
            "Pitch": pitch,
            "Offset H": offset_h_hours,
            "Offset D": offset_d_degrees,
            "Target RA": target_ra_apparent,
            "Target DEC": target_dec_apparent,
            "Actual RA": actual_apparent_ra,
            "Actual DEC": actual_apparent_dec
        })

    df = pd.DataFrame(dataset)
    
    print("\nTable 1: Mount Coordinates and Offsets")
    print(df[["Roll", "Pitch", "Offset H", "Offset D", "LST"]].head())

    print("\nTable 2: Sky Target vs True Physical Mount Position")
    print(df[["Target RA", "Target DEC", "Actual RA", "Actual DEC", "LST"]].head())

    # save to csv
    df.to_csv("~/Documents/synthetic_tpoint_data.dat", sep=" ", index=False)
    print("\nsaved to dat")

if __name__ == "__main__":
    main()



