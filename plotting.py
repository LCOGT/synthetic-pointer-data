import matplotlib.pyplot as mat
import pandas as pd
import os
import numpy as np
from syntheticPoint import IH, ID, CH, NP, MA, ME, TF, PHH, PDD, ARCSEC_TO_RAD, PHI


data = pd.read_csv(os.path.expanduser("~/Documents/synthetic-pointer-data/residuals_data.csv"), sep=" ")
data_old = pd.read_csv(os.path.expanduser("~/Documents/synthetic-pointer-data v.alpha/residuals_data.csv"), sep=" ")
print(data.head())

RAD_TO_ARCSEC = np.degrees(1) * 3600
DEG_TO_ARCSEC = 3600

delta_h_terms = ["ch", "ih", "mah", "meh", "nP", "tfh", "phh"]
delta_d_terms = ["id", "mad", "med", "tfd", "pdd"]

for col in delta_h_terms + delta_d_terms:
    data[col] = data[col] * RAD_TO_ARCSEC

roll = data["roll_hours"]
pitch = data["pitch_deg"]
residual_ra = data["residual_ra"] * DEG_TO_ARCSEC
residual_dec = data["residual_dec"] * DEG_TO_ARCSEC

roll_rad_new = np.radians(data["roll_hours"] * 15.0)
pitch_rad_new = np.radians(data["pitch_deg"])
sin_roll_new = np.sin(roll_rad_new)
cos_roll_new = np.cos(roll_rad_new)
sin_pitch_new = np.sin(pitch_rad_new)
cos_pitch_new = np.cos(pitch_rad_new)

total_offset_roll = data["total_offset_roll"] * DEG_TO_ARCSEC
total_offset_pitch = data["total_offset_pitch"] * DEG_TO_ARCSEC

# Reference text block shared across all figures
ref_text = (
    "TPoint Constants (arcsec):\n"
    f"  IH  = {IH/ARCSEC_TO_RAD:+.1f}\"\n"
    f"  ID  = {ID/ARCSEC_TO_RAD:+.1f}\"\n"
    f"  CH  = {CH/ARCSEC_TO_RAD:+.1f}\"\n"
    f"  NP  = {NP/ARCSEC_TO_RAD:+.1f}\"\n"
    f"  MA  = {MA/ARCSEC_TO_RAD:+.1f}\"\n"
    f"  ME  = {ME/ARCSEC_TO_RAD:+.1f}\"\n"
    f"  TF  = {TF/ARCSEC_TO_RAD:+.1f}\"\n"
    f"  PHH = {PHH/ARCSEC_TO_RAD:+.1f}\"\n"
    f"  PDD = {PDD/ARCSEC_TO_RAD:+.1f}\"\n"
    f"  PHI = {np.degrees(PHI):+.4f} deg\n"
    "\n"
    "Delta-H (RA) Terms:\n"
    "  ch  = CH / cos(pitch)\n"
    "  ih  = IH\n"
    "  mah = -MA * cos(roll) * tan(pitch)\n"
    "  meh = ME * sin(roll) * tan(pitch)\n"
    "  nP  = NP * tan(pitch)\n"
    "  tfh = TF * cos(PHI) * sin(roll) / cos(pitch)\n"
    "  phh = PHH * roll\n"
    "\n"
    "Delta-D (Dec) Terms:\n"
    "  id  = ID\n"
    "  mad = MA * sin(roll)\n"
    "  med = ME * cos(roll)\n"
    "  tfd = TF * (cos(PHI)*cos(roll)*sin(pitch)\n"
    "            - sin(PHI)*cos(pitch))\n"
    "  pdd = PDD * pitch\n"
)


# PLOT 1: Residuals vs Roll and Pitch
fig, axes = mat.subplots(2, 2, figsize=(16, 8))
fig.suptitle("1. Residuals vs. Roll and Pitch\n(year, month, day, hour, minute, second, lst_hours, obs_ra_deg, obs_dec_deg, lat_deg, lon_deg, elevation_m)")

axes[0,0].scatter(roll, residual_ra, s=5, alpha=0.5)
axes[0,0].axhline(0, color='red', linewidth=1)
axes[0,0].set_xlabel("Roll (hours)")
axes[0,0].set_ylabel("Residual RA (arcsec)")

axes[0,1].scatter(pitch, residual_ra, s=5, alpha=0.5)
axes[0,1].axhline(0, color='red', linewidth=1)
axes[0,1].set_xlabel("Pitch (deg)")
axes[0,1].set_ylabel("Residual RA (arcsec)")

axes[1,0].scatter(roll, residual_dec, s=5, alpha=0.5)
axes[1,0].axhline(0, color='red', linewidth=1)
axes[1,0].set_xlabel("Roll (hours)")
axes[1,0].set_ylabel("Residual Dec (arcsec)")

axes[1,1].scatter(pitch, residual_dec, s=5, alpha=0.5)
axes[1,1].axhline(0, color='red', linewidth=1)
axes[1,1].set_xlabel("Pitch (deg)")
axes[1,1].set_ylabel("Residual Dec (arcsec)")

mat.tight_layout(rect=[0, 0, 0.78, 1])
fig.text(0.80, 0.5, ref_text, fontsize=7, verticalalignment='center',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
mat.savefig(os.path.expanduser("~/Documents/synthetic-pointer-data/residual_plots_1.png"), dpi=150, bbox_inches='tight')


# PLOT 2a: Delta-H TPoint terms vs Roll and Pitch
fig, axes = mat.subplots(8, 2, figsize=(16, 32))
fig.suptitle("2a. Tpoint Terms Contributing to ∆roll vs Roll and Pitch")

for i, term in enumerate(delta_h_terms):
    axes[i,0].scatter(roll, data[term], s=5, alpha=0.5)
    axes[i,0].axhline(0, color='red', linewidth=1)
    axes[i,0].set_xlabel("Roll (hours)")
    axes[i,0].set_ylabel(f"{term} (arcsec)")

    axes[i,1].scatter(pitch, data[term], s=5, alpha=0.5)
    axes[i,1].axhline(0, color='red', linewidth=1)
    axes[i,1].set_xlabel("Pitch (deg)")
    axes[i,1].set_ylabel(f"{term} (arcsec)")

axes[7,0].scatter(roll, total_offset_roll, s=5, alpha=0.5)
axes[7,0].axhline(0, color='red', linewidth=1)
axes[7,0].set_xlabel("Roll (hours)")
axes[7,0].set_ylabel("Total Roll Offset (arcsec)")

axes[7,1].scatter(pitch, total_offset_roll, s=5, alpha=0.5)
axes[7,1].axhline(0, color='red', linewidth=1)
axes[7,1].set_xlabel("Pitch (deg)")
axes[7,1].set_ylabel("Total Roll Offset (arcsec)")

mat.tight_layout(rect=[0, 0, 0.78, 1])
fig.text(0.80, 0.5, ref_text, fontsize=7, verticalalignment='center',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
mat.savefig(os.path.expanduser("~/Documents/synthetic-pointer-data/residual_plots_2a.png"), dpi=150, bbox_inches='tight')


# PLOT 2b: Delta-D TPoint terms vs Roll and Pitch
fig, axes = mat.subplots(6, 2, figsize=(16, 24))
fig.suptitle("2b. Tpoint Terms Contributing to ∆pitch vs Roll and Pitch")

for i, term in enumerate(delta_d_terms):
    axes[i,0].scatter(roll, data[term], s=5, alpha=0.5)
    axes[i,0].axhline(0, color='red', linewidth=1)
    axes[i,0].set_xlabel("Roll (hours)")
    axes[i,0].set_ylabel(f"{term} (arcsec)")

    axes[i,1].scatter(pitch, data[term], s=5, alpha=0.5)
    axes[i,1].axhline(0, color='red', linewidth=1)
    axes[i,1].set_xlabel("Pitch (deg)")
    axes[i,1].set_ylabel(f"{term} (arcsec)")

axes[5,0].scatter(roll, total_offset_pitch, s=5, alpha=0.5)
axes[5,0].axhline(0, color='red', linewidth=1)
axes[5,0].set_xlabel("Roll (hours)")
axes[5,0].set_ylabel("Total Pitch Offset (arcsec)")

axes[5,1].scatter(pitch, total_offset_pitch, s=5, alpha=0.5)
axes[5,1].axhline(0, color='red', linewidth=1)
axes[5,1].set_xlabel("Pitch (deg)")
axes[5,1].set_ylabel("Total Pitch Offset (arcsec)")

mat.tight_layout(rect=[0, 0, 0.78, 1])
fig.text(0.80, 0.5, ref_text, fontsize=7, verticalalignment='center',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
mat.savefig(os.path.expanduser("~/Documents/synthetic-pointer-data/residual_plots_2b.png"), dpi=150, bbox_inches='tight')


# PLOT 3a: Residual RA and Dec vs Delta-H TPoint terms
fig, axes = mat.subplots(7, 2, figsize=(16, 24))
fig.suptitle("3a. Residual RA vs. Tpoint terms")

for i, term in enumerate(delta_h_terms):
    axes[i,0].scatter(data[term], residual_ra, s=5, alpha=0.5)
    axes[i,0].axhline(0, color='red', linewidth=1)
    axes[i,0].set_xlabel(f"{term} (arcsec)")
    axes[i,0].set_ylabel("Residual RA (arcsec)")

    axes[i,1].scatter(data[term], residual_dec, s=5, alpha=0.5)
    axes[i,1].axhline(0, color='red', linewidth=1)
    axes[i,1].set_xlabel(f"{term} (arcsec)")
    axes[i,1].set_ylabel("Residual Dec (arcsec)")

mat.tight_layout(rect=[0, 0, 0.78, 1])
fig.text(0.80, 0.5, ref_text, fontsize=7, verticalalignment='center',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
mat.savefig(os.path.expanduser("~/Documents/synthetic-pointer-data/residual_plots_3a.png"), dpi=150, bbox_inches='tight')


# PLOT 3b: Residual RA and Dec vs Delta-D TPoint terms
fig, axes = mat.subplots(5, 2, figsize=(16, 20))
fig.suptitle("3b. Residual Dec vs. Tpoint terms")

for i, term in enumerate(delta_d_terms):
    axes[i,0].scatter(data[term], residual_ra, s=5, alpha=0.5)
    axes[i,0].axhline(0, color='red', linewidth=1)
    axes[i,0].set_xlabel(f"{term} (arcsec)")
    axes[i,0].set_ylabel("Residual RA (arcsec)")

    axes[i,1].scatter(data[term], residual_dec, s=5, alpha=0.5)
    axes[i,1].axhline(0, color='red', linewidth=1)
    axes[i,1].set_xlabel(f"{term} (arcsec)")
    axes[i,1].set_ylabel("Residual Dec (arcsec)")

mat.tight_layout(rect=[0, 0, 0.78, 1])
fig.text(0.80, 0.5, ref_text, fontsize=7, verticalalignment='center',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
mat.savefig(os.path.expanduser("~/Documents/synthetic-pointer-data/residual_plots_3b.png"), dpi=150, bbox_inches='tight')

# PLOT 4: Before/After comparison — old vs new model residuals vs roll
roll_old = data_old["roll_hours"]
residual_ra_old = data_old["residual_ra"] * DEG_TO_ARCSEC
residual_dec_old = data_old["residual_dec"] * DEG_TO_ARCSEC

fig, axes = mat.subplots(2, 2, figsize=(16, 8))
fig.suptitle("4. Before vs After: Residuals vs Roll (old model vs trig-feature model)")

# Shared y-limits for fair comparison
ra_ylim = (
    min(residual_ra_old.min(), residual_ra.min()),
    max(residual_ra_old.max(), residual_ra.max())
)
dec_ylim = (
    min(residual_dec_old.min(), residual_dec.min()),
    max(residual_dec_old.max(), residual_dec.max())
)

axes[0,0].scatter(roll_old, residual_ra_old, s=5, alpha=0.5, color='gray')
axes[0,0].axhline(0, color='red', linewidth=1)
axes[0,0].set_xlabel("Roll (hours)")
axes[0,0].set_ylabel("Residual RA (arcsec)")
axes[0,0].set_title("Old model")
axes[0,0].set_ylim(ra_ylim)

axes[0,1].scatter(roll, residual_ra, s=5, alpha=0.5, color='green')
axes[0,1].axhline(0, color='red', linewidth=1)
axes[0,1].set_xlabel("Roll (hours)")
axes[0,1].set_ylabel("Residual RA (arcsec)")
axes[0,1].set_title("New model (trig features)")
axes[0,1].set_ylim(ra_ylim)

axes[1,0].scatter(roll_old, residual_dec_old, s=5, alpha=0.5, color='gray')
axes[1,0].axhline(0, color='red', linewidth=1)
axes[1,0].set_xlabel("Roll (hours)")
axes[1,0].set_ylabel("Residual Dec (arcsec)")
axes[1,0].set_title("Old model")
axes[1,0].set_ylim(dec_ylim)

axes[1,1].scatter(roll, residual_dec, s=5, alpha=0.5, color='green')
axes[1,1].axhline(0, color='red', linewidth=1)
axes[1,1].set_xlabel("Roll (hours)")
axes[1,1].set_ylabel("Residual Dec (arcsec)")
axes[1,1].set_title("New model (trig features)")
axes[1,1].set_ylim(dec_ylim)

mat.tight_layout(rect=[0, 0, 0.78, 1])
fig.text(0.80, 0.5, ref_text, fontsize=7, verticalalignment='center',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
mat.savefig(os.path.expanduser("~/Documents/synthetic-pointer-data/residual_plots_4_comparison.png"), dpi=150, bbox_inches='tight')


# PLOT 5: Histogram comparison of residual magnitudes — old vs new
fig, axes = mat.subplots(1, 2, figsize=(16, 6))
fig.suptitle("5. Residual Magnitude Distribution: Old vs New Model")

axes[0].hist(np.abs(residual_ra_old), bins=50, alpha=0.5, label="Old model", color='gray')
axes[0].hist(np.abs(residual_ra), bins=50, alpha=0.5, label="New model (trig features)", color='green')
axes[0].set_xlabel("|Residual RA| (arcsec)")
axes[0].set_ylabel("Count")
axes[0].set_title("RA Residual Magnitude")
axes[0].legend()

axes[1].hist(np.abs(residual_dec_old), bins=50, alpha=0.5, label="Old model", color='gray')
axes[1].hist(np.abs(residual_dec), bins=50, alpha=0.5, label="New model (trig features)", color='green')
axes[1].set_xlabel("|Residual Dec| (arcsec)")
axes[1].set_ylabel("Count")
axes[1].set_title("Dec Residual Magnitude")
axes[1].legend()

mat.tight_layout(rect=[0, 0, 0.78, 1])
fig.text(0.80, 0.5, ref_text, fontsize=7, verticalalignment='center',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
mat.savefig(os.path.expanduser("~/Documents/synthetic-pointer-data/residual_plots_5_histogram.png"), dpi=150, bbox_inches='tight')

# PLOT 6: Residual RA and Dec vs new trig features
trig_features = {
    "sin_roll": sin_roll_new,
    "cos_roll": cos_roll_new,
    "sin_pitch": sin_pitch_new,
    "cos_pitch": cos_pitch_new,
}

fig, axes = mat.subplots(4, 2, figsize=(16, 16))
fig.suptitle("6. Residual RA and Dec vs Trig Features (new model inputs)")

for i, (name, values) in enumerate(trig_features.items()):
    axes[i,0].scatter(values, residual_ra, s=5, alpha=0.5)
    axes[i,0].axhline(0, color='red', linewidth=1)
    axes[i,0].set_xlabel(name)
    axes[i,0].set_ylabel("Residual RA (arcsec)")

    axes[i,1].scatter(values, residual_dec, s=5, alpha=0.5)
    axes[i,1].axhline(0, color='red', linewidth=1)
    axes[i,1].set_xlabel(name)
    axes[i,1].set_ylabel("Residual Dec (arcsec)")

mat.tight_layout(rect=[0, 0, 0.78, 1])
fig.text(0.80, 0.5, ref_text, fontsize=7, verticalalignment='center',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
mat.savefig(os.path.expanduser("~/Documents/synthetic-pointer-data/residual_plots_6_trig.png"), dpi=150, bbox_inches='tight')


mat.show()