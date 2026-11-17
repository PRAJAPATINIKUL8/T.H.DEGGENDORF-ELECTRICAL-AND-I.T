# CT Image Spatial Resolution Analysis Using Python

**Nikul Prajapati | Technische Hochschule Deggendorf (THD)**

**Course:** Special Topics of Contactless Sensor Systems

## Results

- Input image analyzed at native resolution: 1301 × 1301 pixels.
- Pixel pitch: 2.143 µm = 0.002143 mm; Nyquist frequency: 233.32 lp/mm.
- Valid measured patterns: **29 / 31**; frequency range: **6.67–100.69 lp/mm**; CTF contrast range: **0.102–0.465**.
- Weighted pooled fit: A = 0.3778 ± 0.0064; σf = 96.16 ± 2.6 lp/mm; R² = 0.313; RMSE = 0.0788.

σf is the **frequency-domain standard deviation**. A is the extrapolated zero-frequency contrast; DC was not directly measured.

## Measurement and fitting method

The original PNG is analyzed at native resolution. ROIs use editable half-open pixel coordinates in `mtf_analysis.py`; H04 and V04 were tightened after visual inspection of the original pixels. V04 was restricted to its vertical-bar patch, excluding the separate horizontal-bar patch immediately above it. For each ROI, parallel pixel lines are averaged to form a profile perpendicular to the stripes. Period is estimated from the profile's windowed Fourier peak, then converted using the measured pixel pitch.

- `Period_mm = Period_pixels × 0.002143 mm/pixel`
- `f (lp/mm) = 1 / Period_mm`
- `C = (Ibright − Idark) / (Ibright + Idark)`
- `C(f) = A exp[−f² / (2σf²)]`; normalized response `C(f)/A`.

The measured stripes are bars, so their Michelson contrast is a **contrast transfer function (CTF)**. It is not a fully corrected sinusoidal MTF: no square-wave harmonic conversion was applied. The normalized Gaussian is an MTF estimate based on measured CTF.

Weighted fitting minimizes residuals scaled by the contrast standard error from the parallel subprofiles (with a 0.005 floor); unweighted fitting gives each pattern equal weight. Both R² and RMSE are calculated on the unweighted contrast residuals.

## Weighted versus unweighted Gaussian fits

| Fit group | Weighting | N | A (extrapolated DC) | σf (lp/mm) | R² | RMSE |
|---|---|---:|---:|---:|---:|---:|
| all orientations | weighted | 29 | 0.3778 ± 0.0064 | 96.16 ± 2.6 | 0.313 | 0.0788 |
| all orientations | unweighted | 29 | 0.3767 ± 0.019 | 64.17 ± 5.4 | 0.657 | 0.0557 |
| vertical | weighted | 14 | 0.3685 ± 0.0086 | 79.79 ± 3.7 | 0.552 | 0.0603 |
| vertical | unweighted | 14 | 0.3717 ± 0.026 | 63.02 ± 7.7 | 0.672 | 0.0516 |
| horizontal | weighted | 15 | 0.3986 ± 0.01 | 96.17 ± 3.2 | 0.199 | 0.0890 |
| horizontal | unweighted | 15 | 0.3823 ± 0.028 | 64.95 ± 7.9 | 0.653 | 0.0586 |

For the pooled fits, unweighted minus weighted is Δσf = -31.985 lp/mm (-33.26%) and ΔA = -0.0012. The weighted pooled result is used for the normalized plot. The weighted fit has R² = 0.313, while the unweighted fit has R² = 0.657; the strong weighting dependence and weak weighted-fit R² mean that neither σf should be treated as a robust, unique system-resolution estimate.

Formal parameter standard errors are conditional on this Gaussian model and the stated weights. They do not include systematic ROI-boundary uncertainty, uncertainty in pixel pitch, finite pattern geometry, or the CTF-to-MTF approximation. The subprofile spread captures within-ROI variability, not changes caused by moving an ROI boundary. Normalized points can exceed 1 because an individual measured CTF can exceed the fitted extrapolated amplitude A; this is fit/target variability, not evidence of physical MTF above unity.

## Measured patterns

| ROI | Pattern | Orientation | Period (px) | Period (mm) | Frequency (lp/mm) | Bright | Dark | Contrast ± SD |
|---|---|---|---:|---:|---:|---:|---:|---:|
| V01 | Large vertical bars, upper left | vertical | 69.92 | 0.14983 | 6.67 | 214.3 | 103.6 | 0.348 ± 0.027 |
| V02 | Medium vertical bars, left center | vertical | 46.62 | 0.09990 | 10.01 | 210.0 | 84.5 | 0.426 ± 0.034 |
| V03 | Fine vertical bars, lower left | vertical | 23.22 | 0.04976 | 20.10 | 209.2 | 79.9 | 0.447 ± 0.091 |
| H01 | Large horizontal bars, upper right | horizontal | 69.47 | 0.14887 | 6.72 | 213.4 | 91.4 | 0.400 ± 0.042 |
| H02 | Medium horizontal bars, right center | horizontal | 46.48 | 0.09961 | 10.04 | 213.0 | 77.9 | 0.465 ± 0.055 |
| H03 | Fine horizontal bars, right lower | horizontal | 23.07 | 0.04944 | 20.23 | 182.5 | 76.7 | 0.408 ± 0.034 |
| H04 | Small horizontal bars, upper group 1 | horizontal | 4.65 | 0.00996 | 100.40 | 214.9 | 131.8 | 0.240 ± 0.013 |
| H05 | Small horizontal bars, upper group 2 | horizontal | 7.13 | 0.01529 | 65.42 | 221.8 | 154.4 | 0.179 ± 0.091 |
| H06 | Small horizontal bars, upper group 3 | horizontal | 9.29 | 0.01990 | 50.25 | 216.3 | 137.1 | 0.224 ± 0.114 |
| H07 | Small horizontal bars, upper group 4 | horizontal | 11.64 | 0.02493 | 40.11 | 211.9 | 113.3 | 0.303 ± 0.166 |
| H08 | Small horizontal bars, upper group 5 | horizontal | 13.99 | 0.02997 | 33.37 | 207.7 | 125.6 | 0.246 ± 0.140 |
| V04 | Small vertical bars, upper group 1 | vertical | 4.63 | 0.00993 | 100.69 | 217.6 | 140.2 | 0.216 ± 0.029 |
| V05 | Small vertical bars, upper group 2 | vertical | 6.96 | 0.01491 | 67.08 | 222.7 | 157.8 | 0.170 ± 0.116 |
| V06 | Small vertical bars, upper group 3 | vertical | 9.34 | 0.02002 | 49.95 | 217.2 | 139.8 | 0.217 ± 0.127 |
| V07 | Small vertical bars, upper group 4 | vertical | 11.53 | 0.02472 | 40.46 | 213.7 | 112.3 | 0.311 ± 0.068 |
| V08 | Small vertical bars, upper group 5 | vertical | 13.93 | 0.02985 | 33.51 | 210.4 | 111.5 | 0.307 ± 0.079 |
| V09 | Vertical bars, middle test group 1 | vertical | 4.72 | 0.01012 | 98.82 | 197.4 | 155.5 | 0.119 ± 0.047 |
| V10 | Vertical bars, middle test group 2 | vertical | 7.10 | 0.01521 | 65.75 | 204.1 | 135.3 | 0.203 ± 0.069 |
| V11 | Vertical bars, middle test group 3 | vertical | 9.25 | 0.01982 | 50.44 | 205.4 | 127.3 | 0.235 ± 0.081 |
| V12 | Vertical bars, middle test group 4 | vertical | 11.58 | 0.02482 | 40.29 | 206.9 | 127.2 | 0.238 ± 0.082 |
| V13 | Vertical bars, middle test group 5 | vertical | 13.89 | 0.02976 | 33.60 | 207.9 | 113.1 | 0.295 ± 0.114 |
| H09 | Horizontal bars, middle test group 1 | horizontal | 4.65 | 0.00998 | 100.25 | 195.0 | 158.8 | 0.102 ± 0.087 |
| H10 | Horizontal bars, middle test group 2 | horizontal | 7.01 | 0.01501 | 66.61 | 200.2 | 138.9 | 0.181 ± 0.162 |
| H11 | Horizontal bars, middle test group 3 | horizontal | 9.34 | 0.02002 | 49.95 | 200.9 | 128.4 | 0.220 ± 0.152 |
| H12 | Horizontal bars, middle test group 4 | horizontal | 11.85 | 0.02540 | 39.38 | 201.7 | 105.7 | 0.313 ± 0.183 |
| H13 | Horizontal bars, middle test group 5 | horizontal | 14.18 | 0.03038 | 32.91 | 204.3 | 91.0 | 0.384 ± 0.112 |
| V14 | Small vertical bars, lower left group | vertical | 11.81 | 0.02531 | 39.50 | 200.8 | 101.7 | 0.328 ± 0.218 |
| H15 | Fine horizontal bars, lower right left group | horizontal | 4.84 | 0.01036 | 96.50 | 172.3 | 118.3 | 0.186 ± 0.070 |
| H16 | Fine horizontal bars, lower right group | horizontal | 11.82 | 0.02534 | 39.47 | 167.4 | 86.8 | 0.317 ± 0.062 |

## Exclusions and limitations

Printed markings, dot/ring patterns, borders, and nonperiodic background were not treated as stripe ROIs. Frequency is measured from the pixel data, not read from printed labels. Candidate ROIs with fewer than three cycles, weak periodicity, above-Nyquist frequency, or insufficient sampled stripe interiors are excluded:

- **H14 — Small horizontal bars, lower left group:** Only 2.50 cycles in ROI; at least 3.0 are required.
- **V15 — Fine vertical bars, lower right left group:** Fundamental periodicity R² = 0.160 is below 0.20; the profile is not sufficiently periodic.

Finite ROI length, slight stripe tilt, image noise, and uncertain bar duty cycle also limit interpretation. `measurements.csv` includes all configured ROIs and exclusion reasons; `fit_results.csv` includes weighted and unweighted pooled and orientation-specific fits.

## Figures

- `intensity_profiles.png` — averaged grayscale profiles.
- `representative_intensity_profiles.png` — selected profiles at low, medium, and high spatial frequencies.
- `contrast_transfer_vs_spatial_frequency.png` — raw measured CTF.
- `normalized_mtf_estimate_and_gaussian_fits.png` — normalized CTF points and Gaussian fits.
- `../CT_Image_Spatial_Resolution_Analysis_Report.pdf` — image-free multipage report.

The source CT image and its ROI overview are not included in this repository because redistribution permission could not be established. The PDF report contains plots, measurements, methods, and limitations, but does not embed the source image.

## Development and AI Assistance

I reviewed the project requirements, supplied the input image, defined the analysis requirements, and reviewed the outputs. GitHub Copilot assisted with the Python implementation, numerical calculations, plotting, and documentation. The project was tested using the original supplied image.
