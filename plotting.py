import matplotlib.pyplot as mat
import pandas as pd
import os

data = pd.read_csv(os.path.expanduser("~/Documents/synthetic-pointer-data/residuals_data.csv"), sep=" ")
print(data.head())

roll = data["roll_hours"]
pitch = data["pitch_deg"]
residual_ra = data["residual_ra"]
residual_dec = data["residual_dec"]

fig, axes = mat.subplots(2, 2, figsize = (12, 8))
fig.suptitle("Residuals vs. Roll and Pitch")


#top left graph: residual ra vs. roll
axes[0,0].scatter(roll, residual_ra, s=5, alpha=0.5)
axes[0,0].axhline(0, color='red', linewidth=1)
axes[0,0].set_xlabel("Roll (hours)")
axes[0,0].set_ylabel("Residual RA (deg)")

#top right: residual ra vs. pitch
axes[0,1].scatter(pitch, residual_ra, s=5, alpha=0.5)
axes[0,1].axhline(0, color='red', linewidth=1)
axes[0,1].set_xlabel("Pitch (deg)")
axes[0,1].set_ylabel("Residual RA (deg)")

#bottom left: residual dec vs. roll
axes[1,0].scatter(roll, residual_dec, s=5, alpha=0.5)
axes[1,0].axhline(0, color='red', linewidth=1)
axes[1,0].set_xlabel("Roll (hours)")
axes[1,0].set_ylabel("Residual Dec (deg)")

#bottom right: residual dec vs. pitch
axes[1,1].scatter(pitch, residual_dec, s=5, alpha=0.5)
axes[1,1].axhline(0, color='red', linewidth=1)
axes[1,1].set_xlabel("Pitch (deg)")
axes[1,1].set_ylabel("Residual Dec (deg)")

mat.tight_layout()
mat.savefig(os.path.expanduser("~/Documents/synthetic-pointer-data/residual_plots.png"), dpi=150)
mat.show()