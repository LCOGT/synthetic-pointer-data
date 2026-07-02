import matplotlib.pyplot as mat
import pandas as pd
import os
import numpy as np
from syntheticPoint import IH, ID, CH, NP, MA, ME, TF, PHH, PDD, ARCSEC_TO_RAD, PHI


data = pd.read_csv(os.path.expanduser("~/Documents/synthetic-pointer-data/residuals_data.csv"), sep=" ")
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


mat.show()