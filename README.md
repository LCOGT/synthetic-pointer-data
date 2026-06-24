# Synthetic Telescope Pointing Data Generator

### Overview and Workflow

The script simulates data-driven telescope telemetry modeled after the telescope in Haleakala, Hawaii (ogg.clma.2m0a). It is designed to create a baseline dataset for training the machine learning model presented at SPIE 2026 researching data-driven acquisition offset correction for telescope networks to validate its accuracy against synthetic Tpoint-based calculations. The generator executes according to the following steps:

1. **Coordinate Frame Transformation:** The script picks a dynamic observation timestamp and a random local sidereal time (LST), then translates randomly generated raw star coordinates from the static celestial ICRS frame into the local apparent Celestial Intermediate Reference System (CIRS) using `astropy`.
2. **Mount Frame Projection:** It calculates local hour angle (Roll) and declination (Pitch) parameters to project the target star's position onto an equatorial telescope mount structure.
3. **Horizon Filtering:** An automated horizon check filters out invalid observations. Any targets falling outside a realistic tracking window (Hour Angles beyond $\pm 5$ hours) are dropped to mimic actual observing conditions.
4. **Mechanical Error Injection:** The remaining valid coordinates are processed through traditional TPoint mathematical pointing formulations to inject mechanical imperfections—such as tube flexure, non-perpendicularity, and encoder index errors.
5. **Offset and Actual Coordinate Derivation:** The script computes the resulting angular offsets (`Offset H`, `Offset D`) and outputs the true physical coordinates where the telescope mechanics would actually land (`Actual RA`, `Actual DEC`).
6. **Fixed-Width Export:** The final output is written to a cleanly aligned, fixed-width space-separated text file (`synthetic_tpoint_data.dat`), matching the strict preprocessing schema required by modern astronomical pipeline utilities.
