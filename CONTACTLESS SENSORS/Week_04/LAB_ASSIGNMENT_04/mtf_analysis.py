"""Analyze CT image spatial resolution with bar-pattern contrast and Gaussian fits."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import textwrap
import warnings

import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit, minimize_scalar


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
PROJECT_TITLE = "CT Image Spatial Resolution Analysis Using Python"
IMAGE_CANDIDATES = (
    PROJECT_DIR / "Horizontal_QRM_Micro_tomolibri (2).png",
    PROJECT_DIR / "Horizontal_QRM_Micro_tomolibri.png",
)
EXPECTED_SHAPE = (1301, 1301)
PIXEL_PITCH_UM = 2.143
PIXEL_PITCH_MM = PIXEL_PITCH_UM / 1000.0
NYQUIST_LP_PER_MM = 1.0 / (2.0 * PIXEL_PITCH_MM)
MIN_CYCLES = 3.0
MIN_PERIODIC_R_SQUARED = 0.20

# Pixel coordinates are half-open: x0 <= x < x1 and y0 <= y < y1.
# "vertical" varies across x; "horizontal" varies across y.
ROI_CONFIG = [
    ("V01", "Large vertical bars, upper left", (112, 137, 458, 489), "vertical"),
    ("V02", "Medium vertical bars, left center", (121, 555, 355, 793), "vertical"),
    ("V03", "Fine vertical bars, lower left", (130, 832, 250, 960), "vertical"),
    ("H01", "Large horizontal bars, upper right", (800, 126, 1117, 455), "horizontal"),
    ("H02", "Medium horizontal bars, right center", (916, 508, 1115, 731), "horizontal"),
    ("H03", "Fine horizontal bars, right lower", (918, 768, 1158, 875), "horizontal"),
    ("H04", "Small horizontal bars, upper group 1", (587, 160, 657, 184), "horizontal"),
    ("H05", "Small horizontal bars, upper group 2", (574, 203, 665, 251), "horizontal"),
    ("H06", "Small horizontal bars, upper group 3", (574, 268, 665, 320), "horizontal"),
    ("H07", "Small horizontal bars, upper group 4", (574, 335, 665, 394), "horizontal"),
    ("H08", "Small horizontal bars, upper group 5", (574, 408, 665, 487), "horizontal"),
    ("V04", "Small vertical bars, upper group 1", (679, 157, 739, 182), "vertical"),
    ("V05", "Small vertical bars, upper group 2", (670, 199, 750, 250), "vertical"),
    ("V06", "Small vertical bars, upper group 3", (670, 267, 750, 320), "vertical"),
    ("V07", "Small vertical bars, upper group 4", (670, 334, 750, 395), "vertical"),
    ("V08", "Small vertical bars, upper group 5", (670, 408, 750, 487), "vertical"),
    ("V09", "Vertical bars, middle test group 1", (489, 604, 528, 688), "vertical"),
    ("V10", "Vertical bars, middle test group 2", (532, 604, 586, 688), "vertical"),
    ("V11", "Vertical bars, middle test group 3", (590, 604, 654, 688), "vertical"),
    ("V12", "Vertical bars, middle test group 4", (662, 604, 738, 688), "vertical"),
    ("V13", "Vertical bars, middle test group 5", (742, 604, 819, 688), "vertical"),
    ("H09", "Horizontal bars, middle test group 1", (489, 695, 528, 775), "horizontal"),
    ("H10", "Horizontal bars, middle test group 2", (532, 695, 586, 775), "horizontal"),
    ("H11", "Horizontal bars, middle test group 3", (590, 695, 654, 775), "horizontal"),
    ("H12", "Horizontal bars, middle test group 4", (662, 695, 738, 775), "horizontal"),
    ("H13", "Horizontal bars, middle test group 5", (742, 695, 819, 775), "horizontal"),
    ("H14", "Small horizontal bars, lower left group", (291, 828, 356, 878), "horizontal"),
    ("V14", "Small vertical bars, lower left group", (291, 883, 356, 962), "vertical"),
    ("H15", "Fine horizontal bars, lower right left group", (918, 894, 1003, 922), "horizontal"),
    ("V15", "Fine vertical bars, lower right left group", (918, 923, 1003, 957), "vertical"),
    ("H16", "Fine horizontal bars, lower right group", (1036, 893, 1163, 958), "horizontal"),
]


@dataclass(frozen=True)
class ROI:
    roi_id: str
    pattern: str
    bounds: tuple[int, int, int, int]
    orientation: str


def add_student_header(fig: plt.Figure, figure_title: str) -> None:
    fig.text(0.5, 0.965, "Student: NIKUL PRAJAPATI", ha="center", va="top",
             fontsize=13, fontweight="bold", color="#18324a")
    fig.text(
        0.5, 0.932,
        "Technische Hochschule Deggendorf | Special Topics of Contactless Sensor Systems",
        ha="center", va="top", fontsize=9, color="#34495e",
    )
    fig.text(0.5, 0.900, f"{PROJECT_TITLE} — {figure_title}", ha="center",
             va="top", fontsize=10, color="#34495e")
    fig.add_artist(plt.Line2D(
        [0.06, 0.94], [0.875, 0.875], transform=fig.transFigure,
        color="#cbd5df", linewidth=0.8,
    ))


def wrap_pdf_text(text: str, width: int) -> str:
    return "\n".join(
        textwrap.fill(line, width=width) if line else ""
        for line in text.splitlines()
    )


def load_image() -> np.ndarray:
    image_path = next((path for path in IMAGE_CANDIDATES if path.is_file()), None)
    if image_path is None:
        names = " or ".join(path.name for path in IMAGE_CANDIDATES)
        raise FileNotFoundError(
            f"Original CT image not found. Place {names} in {PROJECT_DIR}."
        )

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"OpenCV could not read the PNG file: {image_path}")
    if image.ndim == 2:
        grayscale = image
    elif image.ndim == 3 and image.shape[2] == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape {image.shape} in {image_path}.")

    if grayscale.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Expected the original image to be {EXPECTED_SHAPE[1]} x "
            f"{EXPECTED_SHAPE[0]} pixels; found {grayscale.shape[1]} x "
            f"{grayscale.shape[0]} in {image_path}. No resizing is performed."
        )
    print(f"Image: {image_path.name} | {grayscale.shape[1]} x {grayscale.shape[0]} px")
    return grayscale.astype(np.float64, copy=False)


def validate_rois(image: np.ndarray) -> list[ROI]:
    height, width = image.shape
    rois: list[ROI] = []
    for roi_id, pattern, bounds, orientation in ROI_CONFIG:
        x0, y0, x1, y1 = bounds
        if orientation not in {"vertical", "horizontal"}:
            raise ValueError(f"{roi_id}: orientation must be vertical or horizontal.")
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(
                f"{roi_id}: invalid ROI bounds {bounds} for image "
                f"{width} x {height}; bounds are half-open pixel coordinates."
            )
        if min(x1 - x0, y1 - y0) < 8:
            raise ValueError(f"{roi_id}: ROI is too small to measure reliably.")
        rois.append(ROI(roi_id, pattern, bounds, orientation))
    return rois


def get_roi_profile(image: np.ndarray, roi: ROI) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = roi.bounds
    crop = image[y0:y1, x0:x1]
    if roi.orientation == "vertical":
        return crop.mean(axis=0), np.arange(x0, x1, dtype=float)
    return crop.mean(axis=1), np.arange(y0, y1, dtype=float)


def estimate_period(profile: np.ndarray) -> float:
    """Estimate the fundamental period in pixels using a windowed FFT peak."""
    n = profile.size
    if n < 8:
        raise ValueError("The profile is too short to estimate a stripe period.")

    detrended = profile - np.polyval(np.polyfit(np.arange(n), profile, 1), np.arange(n))
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(detrended * window))
    frequencies = np.fft.rfftfreq(n)
    # Require at least 2.5 cycles in the ROI; single bars and broad shading are not periods.
    valid = (frequencies >= 2.5 / n) & (frequencies <= 0.5)
    if not np.any(valid):
        raise ValueError("No frequency bins are available for this ROI.")
    valid_indices = np.flatnonzero(valid)
    peak_index = int(valid_indices[np.argmax(spectrum[valid])])
    if spectrum[peak_index] <= np.finfo(float).eps:
        raise ValueError("No periodic modulation was detected in this ROI.")

    log_spectrum = np.log(spectrum + np.finfo(float).eps)
    offset = 0.0
    if 0 < peak_index < spectrum.size - 1:
        left, center, right = log_spectrum[peak_index - 1:peak_index + 2]
        denominator = left - 2.0 * center + right
        if abs(denominator) > np.finfo(float).eps:
            offset = float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
    initial_frequency = (peak_index + offset) / n
    bin_width = 1.0 / n
    lower = max(2.5 / n, initial_frequency - bin_width)
    upper = min(0.5, initial_frequency + bin_width)

    sample = np.arange(n, dtype=float)

    def sinusoid_power(frequency: float) -> float:
        angle = 2.0 * np.pi * frequency * sample
        design = np.column_stack(
            (np.ones(n), sample - sample.mean(), np.cos(angle), np.sin(angle))
        )
        coefficients, *_ = np.linalg.lstsq(design, profile, rcond=None)
        return float(coefficients[2] ** 2 + coefficients[3] ** 2)

    if upper > lower:
        refined = minimize_scalar(
            lambda frequency: -sinusoid_power(frequency),
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1e-8},
        )
        if refined.success:
            initial_frequency = float(refined.x)
    if not 0.0 < initial_frequency <= 0.5:
        raise ValueError("Estimated stripe frequency is outside the sampled range.")
    return 1.0 / initial_frequency


def phase_centered_levels(profile: np.ndarray, period_px: float) -> tuple[float, float]:
    """Estimate stripe-interior means from a profile using its fitted fundamental phase."""
    n = profile.size
    sample = np.arange(n, dtype=float)
    angle = 2.0 * np.pi * sample / period_px
    design = np.column_stack(
        (np.ones(n), sample - sample.mean(), np.cos(angle), np.sin(angle))
    )
    coefficients, *_ = np.linalg.lstsq(design, profile, rcond=None)
    peak_phase = np.arctan2(coefficients[3], coefficients[2])
    phase = angle - peak_phase
    central_bright = np.cos(phase) >= 0.7
    central_dark = np.cos(phase) <= -0.7
    if central_bright.sum() < 2 or central_dark.sum() < 2:
        raise ValueError(
            "Too few sampled pixels fall inside bright/dark stripe interiors."
        )
    return float(profile[central_bright].mean()), float(profile[central_dark].mean())


def measure_roi(image: np.ndarray, roi: ROI) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    profile, coordinate = get_roi_profile(image, roi)
    record: dict[str, object] = {
        "roi_id": roi.roi_id,
        "pattern": roi.pattern,
        "orientation": roi.orientation,
        "variation_axis": "x" if roi.orientation == "vertical" else "y",
        "x0": roi.bounds[0],
        "y0": roi.bounds[1],
        "x1": roi.bounds[2],
        "y1": roi.bounds[3],
        "profile_length_px": int(profile.size),
        "cycles_in_roi": np.nan,
        "period_px": np.nan,
        "period_mm": np.nan,
        "frequency_lp_per_mm": np.nan,
        "nyquist_lp_per_mm": NYQUIST_LP_PER_MM,
        "nyquist_status": "not estimated",
        "periodic_r_squared": np.nan,
        "bright_intensity_mean": np.nan,
        "bright_intensity_sd": np.nan,
        "dark_intensity_mean": np.nan,
        "dark_intensity_sd": np.nan,
        "contrast": np.nan,
        "contrast_sd": np.nan,
        "contrast_sem": np.nan,
        "subprofiles_measured": 0,
        "usable": False,
        "exclusion_reason": "",
    }

    try:
        period_px = estimate_period(profile)
    except ValueError as error:
        record["exclusion_reason"] = str(error)
        return record, profile, coordinate

    cycles = profile.size / period_px
    period_mm = period_px * PIXEL_PITCH_MM
    frequency = 1.0 / period_mm
    coordinate = np.arange(profile.size, dtype=float)
    trend = np.polyval(np.polyfit(coordinate, profile, 1), coordinate)
    detrended = profile - trend
    angle = 2.0 * np.pi * coordinate / period_px
    design = np.column_stack(
        (np.ones(profile.size), coordinate - coordinate.mean(), np.cos(angle), np.sin(angle))
    )
    periodic_coefficients, *_ = np.linalg.lstsq(design, profile, rcond=None)
    periodic_residual = profile - design @ periodic_coefficients
    trend_energy = float(np.sum(detrended**2))
    periodic_r_squared = (
        1.0 - float(np.sum(periodic_residual**2)) / trend_energy
        if trend_energy > np.finfo(float).eps else 0.0
    )
    record.update({
        "cycles_in_roi": cycles,
        "period_px": period_px,
        "period_mm": period_mm,
        "frequency_lp_per_mm": frequency,
        "periodic_r_squared": periodic_r_squared,
        "nyquist_status": "above Nyquist" if frequency > NYQUIST_LP_PER_MM else "within Nyquist",
    })

    if cycles < MIN_CYCLES:
        record["exclusion_reason"] = (
            f"Only {cycles:.2f} cycles in ROI; at least {MIN_CYCLES:.1f} are required."
        )
        return record, profile, coordinate
    if periodic_r_squared < MIN_PERIODIC_R_SQUARED:
        record["exclusion_reason"] = (
            f"Fundamental periodicity R² = {periodic_r_squared:.3f} is below "
            f"{MIN_PERIODIC_R_SQUARED:.2f}; the profile is not sufficiently periodic."
        )
        return record, profile, coordinate
    if frequency > NYQUIST_LP_PER_MM:
        record["exclusion_reason"] = (
            f"{frequency:.2f} lp/mm exceeds Nyquist ({NYQUIST_LP_PER_MM:.2f} lp/mm)."
        )
        return record, profile, coordinate

    x0, y0, x1, y1 = roi.bounds
    crop = image[y0:y1, x0:x1]
    parallel_length = crop.shape[0] if roi.orientation == "vertical" else crop.shape[1]
    band_count = min(5, max(1, parallel_length // 6))
    bands = np.array_split(crop, band_count, axis=0 if roi.orientation == "vertical" else 1)
    bright_values: list[float] = []
    dark_values: list[float] = []
    contrast_values: list[float] = []
    for band in bands:
        subprofile = band.mean(axis=0 if roi.orientation == "vertical" else 1)
        try:
            bright, dark = phase_centered_levels(subprofile, period_px)
        except ValueError:
            continue
        if bright + dark <= 0:
            continue
        bright_values.append(bright)
        dark_values.append(dark)
        contrast_values.append((bright - dark) / (bright + dark))

    if not contrast_values:
        record["exclusion_reason"] = (
            "No subprofile had enough sampled bright and dark stripe-interior pixels."
        )
        return record, profile, coordinate

    contrast_array = np.asarray(contrast_values)
    bright_array = np.asarray(bright_values)
    dark_array = np.asarray(dark_values)
    bright_mean = float(bright_array.mean())
    dark_mean = float(dark_array.mean())
    record.update({
        "bright_intensity_mean": bright_mean,
        "bright_intensity_sd": float(bright_array.std(ddof=1)) if bright_array.size > 1 else 0.0,
        "dark_intensity_mean": dark_mean,
        "dark_intensity_sd": float(dark_array.std(ddof=1)) if dark_array.size > 1 else 0.0,
        "contrast": (bright_mean - dark_mean) / (bright_mean + dark_mean),
        "contrast_sd": float(contrast_array.std(ddof=1)) if contrast_array.size > 1 else 0.0,
        "contrast_sem": float(contrast_array.std(ddof=1) / np.sqrt(contrast_array.size))
        if contrast_array.size > 1 else 0.0,
        "subprofiles_measured": int(contrast_array.size),
        "usable": True,
    })
    return record, profile, coordinate


def gaussian_response(
    frequency: np.ndarray | float, amplitude: float, sigma_f: float
) -> np.ndarray | float:
    return amplitude * np.exp(-(np.asarray(frequency) ** 2) / (2.0 * sigma_f**2))


def fit_gaussian(data: pd.DataFrame, name: str, weighted: bool) -> dict[str, object]:
    fit: dict[str, object] = {
        "fit_group": name,
        "weighting": "weighted" if weighted else "unweighted",
        "n_measurements": int(len(data)),
        "amplitude_A": np.nan,
        "amplitude_A_std_error": np.nan,
        "sigma_f_lp_per_mm": np.nan,
        "sigma_f_std_error_lp_per_mm": np.nan,
        "r_squared": np.nan,
        "rmse": np.nan,
        "fit_status": "not fitted: fewer than 3 measurements",
    }
    if len(data) < 3:
        return fit

    frequency = data["frequency_lp_per_mm"].to_numpy(dtype=float)
    contrast = data["contrast"].to_numpy(dtype=float)
    initial_amplitude = float(np.clip(np.max(contrast), 0.01, 1.4))
    initial_sigma = float(max(np.median(frequency), 1.0))
    fit_options: dict[str, object] = {}
    if weighted:
        standard_error = data["contrast_sem"].to_numpy(dtype=float)
        fit_options = {
            "sigma": np.maximum(standard_error, 0.005),
            "absolute_sigma": True,
        }
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        try:
            parameters, covariance = curve_fit(
                gaussian_response,
                frequency,
                contrast,
                p0=(initial_amplitude, initial_sigma),
                bounds=([0.0, 0.01], [1.5, max(float(frequency.max()) * 10.0, 10.0)]),
                maxfev=20_000,
                **fit_options,
            )
        except (RuntimeError, ValueError, OptimizeWarning) as error:
            fit["fit_status"] = f"fit failed: {error}"
            return fit

    predicted = gaussian_response(frequency, *parameters)
    residual = contrast - predicted
    total_sum_squares = float(np.sum((contrast - contrast.mean()) ** 2))
    r_squared = (
        1.0 - float(np.sum(residual**2)) / total_sum_squares
        if total_sum_squares > 0 else np.nan
    )
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    fit.update({
        "amplitude_A": float(parameters[0]),
        "amplitude_A_std_error": float(standard_errors[0]),
        "sigma_f_lp_per_mm": float(parameters[1]),
        "sigma_f_std_error_lp_per_mm": float(standard_errors[1]),
        "r_squared": r_squared,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "fit_status": "fitted",
    })
    return fit


def save_roi_overview(image: np.ndarray, rois: list[ROI], measurements: pd.DataFrame) -> None:
    status_by_id = measurements.set_index("roi_id")["usable"].to_dict()
    fig, ax = plt.subplots(figsize=(12, 12))
    add_student_header(fig, "Measured ROI Overview")
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.06, top=0.80)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    for number, roi in enumerate(rois, start=1):
        x0, y0, x1, y1 = roi.bounds
        usable = bool(status_by_id.get(roi.roi_id, False))
        color = "#176b87" if roi.orientation == "vertical" else "#d97706"
        ax.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color,
            linewidth=1.3 if usable else 0.8,
            linestyle="-" if usable else "--", alpha=0.95 if usable else 0.6,
        ))
        ax.text(
            x0, max(0, y0 - 4), f"{number:02d}", color="white", fontsize=7,
            fontweight="bold", ha="left", va="bottom",
            bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.8, "pad": 1.2},
        )
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_xlabel("Image x coordinate (pixels)")
    ax.set_ylabel("Image y coordinate (pixels)")
    ax.set_title("Editable candidate ROIs; solid = measured, dashed = excluded", pad=10)
    ax.legend(handles=[
        Patch(facecolor="none", edgecolor="#176b87", label="Vertical stripes (x profile)"),
        Patch(facecolor="none", edgecolor="#d97706", label="Horizontal stripes (y profile)"),
    ], loc="lower right", framealpha=0.95)
    fig.savefig(RESULTS_DIR / "roi_overview.png", dpi=300, bbox_inches="tight")


def save_intensity_profiles(
    profiles: dict[str, tuple[np.ndarray, np.ndarray]],
    measurements: pd.DataFrame,
) -> None:
    valid = measurements.loc[measurements["usable"]]
    if valid.empty:
        raise RuntimeError("No ROIs passed the measurement checks; no profile plot can be made.")
    columns = 4
    rows = int(np.ceil(len(valid) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(16, max(8, rows * 2.8)), squeeze=False)
    add_student_header(fig, "Averaged Stripe Intensity Profiles")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.06, top=0.80, hspace=0.55, wspace=0.32)
    for ax, row in zip(axes.flat, valid.itertuples(index=False)):
        profile, coordinate = profiles[row.roi_id]
        ax.plot(coordinate, profile, color="#18324a", linewidth=1.1, label="ROI-averaged profile")
        ax.axhline(row.bright_intensity_mean, color="#d97706", linestyle="--",
                   linewidth=0.9, label="Bright-interior mean")
        ax.axhline(row.dark_intensity_mean, color="#176b87", linestyle=":",
                   linewidth=1.0, label="Dark-interior mean")
        ax.set_title(f"{row.roi_id}: {row.pattern}", fontsize=8)
        ax.set_xlabel(f"{row.variation_axis} position (pixels)", fontsize=8)
        ax.set_ylabel("Grayscale intensity (0–255)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=6, loc="best")
    for ax in axes.flat[len(valid):]:
        ax.remove()
    fig.savefig(RESULTS_DIR / "intensity_profiles.png", dpi=300, bbox_inches="tight")


def save_representative_profiles(
    profiles: dict[str, tuple[np.ndarray, np.ndarray]],
    measurements: pd.DataFrame,
) -> None:
    valid = measurements.loc[measurements["usable"]].set_index("roi_id")
    representative_ids = ("V01", "H01", "V11", "H11", "V04", "H04")
    missing = [roi_id for roi_id in representative_ids if roi_id not in valid.index]
    if missing:
        raise ValueError(f"Representative ROI(s) are missing or unusable: {', '.join(missing)}")

    fig, axes = plt.subplots(3, 2, figsize=(11, 9), squeeze=False)
    add_student_header(fig, "Representative Intensity Profiles Across Spatial Frequencies")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.80, hspace=0.62, wspace=0.28)
    for ax, roi_id in zip(axes.flat, representative_ids):
        row = valid.loc[roi_id]
        profile, coordinate = profiles[roi_id]
        ax.plot(coordinate, profile, color="#18324a", linewidth=1.1, label="Averaged profile")
        ax.axhline(
            row["bright_intensity_mean"], color="#d97706", linestyle="--",
            linewidth=0.9, label="Bright mean",
        )
        ax.axhline(
            row["dark_intensity_mean"], color="#176b87", linestyle=":",
            linewidth=1.0, label="Dark mean",
        )
        ax.set_title(
            f"{roi_id} — {row['frequency_lp_per_mm']:.2f} lp/mm "
            f"({row['orientation']} stripes)",
            fontsize=9,
        )
        ax.set_xlabel(f"Image {row['variation_axis']} position (px)", fontsize=8)
        ax.set_ylabel("Gray level (0–255)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, 0.025), fontsize=8,
    )
    fig.savefig(
        RESULTS_DIR / "representative_intensity_profiles.png",
        dpi=300, bbox_inches="tight",
    )


def save_contrast_plot(measurements: pd.DataFrame) -> None:
    valid = measurements.loc[measurements["usable"]]
    fig, ax = plt.subplots(figsize=(10, 7))
    add_student_header(fig, "Raw Contrast versus Spatial Frequency")
    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.13, top=0.80)
    colors = {"vertical": "#176b87", "horizontal": "#d97706"}
    for orientation, group in valid.groupby("orientation"):
        ax.errorbar(
            group["frequency_lp_per_mm"], group["contrast"],
            yerr=group["contrast_sd"], fmt="o", capsize=3, markersize=6,
            color=colors[orientation], label=f"{orientation.capitalize()} stripes",
        )
    ax.set_xlabel("Spatial frequency (line pairs/mm)")
    ax.set_ylabel("Michelson contrast (CTF)")
    ax.set_title("Measured raw bar-pattern contrast; no DC measurement", pad=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(
        RESULTS_DIR / "contrast_transfer_vs_spatial_frequency.png",
        dpi=300, bbox_inches="tight",
    )


def save_mtf_plot(measurements: pd.DataFrame, fits: list[dict[str, object]]) -> None:
    valid = measurements.loc[measurements["usable"]]
    weighted_fits = [fit for fit in fits if fit["weighting"] == "weighted"]
    pooled_fit = next(
        fit for fit in weighted_fits if fit["fit_group"] == "all orientations"
    )
    pooled_unweighted = next(
        fit for fit in fits
        if fit["fit_group"] == "all orientations" and fit["weighting"] == "unweighted"
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    add_student_header(fig, "Normalized MTF Estimate and Gaussian Fit")
    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.13, top=0.80)
    colors = {"vertical": "#176b87", "horizontal": "#d97706"}
    amplitude = pooled_fit["amplitude_A"]
    if np.isfinite(amplitude) and float(amplitude) > 0:
        for orientation, group in valid.groupby("orientation"):
            ax.errorbar(
                group["frequency_lp_per_mm"], group["contrast"] / float(amplitude),
                yerr=group["contrast_sd"] / float(amplitude), fmt="o", capsize=3,
                markersize=6, color=colors[orientation],
                label=f"{orientation.capitalize()} measured CTF / pooled A",
            )
        maximum_frequency = float(valid["frequency_lp_per_mm"].max())
        fit_frequencies = np.linspace(0.0, maximum_frequency, 500)
        for fit in weighted_fits:
            sigma_f = fit["sigma_f_lp_per_mm"]
            if not np.isfinite(sigma_f) or float(sigma_f) <= 0:
                continue
            if fit["fit_group"] == "all orientations":
                curve_color = "#18324a"
                label = "Pooled normalized Gaussian"
            else:
                curve_color = colors[str(fit["fit_group"])]
                label = f"{str(fit['fit_group']).capitalize()} normalized Gaussian"
            ax.plot(
                fit_frequencies,
                np.exp(-(fit_frequencies**2) / (2.0 * float(sigma_f) ** 2)),
                color=curve_color, linewidth=1.8, linestyle="--" if fit["fit_group"] != "all orientations" else "-",
                label=label,
            )
        unweighted_sigma = float(pooled_unweighted["sigma_f_lp_per_mm"])
        ax.plot(
            fit_frequencies,
            np.exp(-(fit_frequencies**2) / (2.0 * unweighted_sigma**2)),
            color="#7353a6", linewidth=1.6, linestyle=":",
            label="Pooled unweighted Gaussian",
        )
    else:
        ax.text(
            0.5, 0.5, "Gaussian fit unavailable: fewer than three valid measurements",
            transform=ax.transAxes, ha="center", va="center",
        )
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Spatial frequency (line pairs/mm)")
    ax.set_ylabel("Normalized response, C(f) / A")
    ax.set_title(
        f"Gaussian fit: exp[−f²/(2σf²)]; DC is extrapolated, not measured "
        f"(Nyquist = {NYQUIST_LP_PER_MM:.1f} lp/mm)",
        pad=10,
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(
        RESULTS_DIR / "normalized_mtf_estimate_and_gaussian_fits.png",
        dpi=300, bbox_inches="tight",
    )


def format_fit_value(fit: dict[str, object], value: str, error: str, unit: str) -> str:
    result = fit[value]
    uncertainty = fit[error]
    suffix = f" {unit}" if unit else ""
    if not np.isfinite(result):
        return "not fitted"
    if np.isfinite(uncertainty):
        return f"{float(result):.4g} ± {float(uncertainty):.2g}{suffix}"
    return f"{float(result):.4g}{suffix}"


def write_report(measurements: pd.DataFrame, fits: list[dict[str, object]]) -> None:
    valid = measurements.loc[measurements["usable"]].copy()
    excluded = measurements.loc[~measurements["usable"]].copy()
    pooled_weighted = next(
        fit for fit in fits
        if fit["fit_group"] == "all orientations" and fit["weighting"] == "weighted"
    )
    pooled_unweighted = next(
        fit for fit in fits
        if fit["fit_group"] == "all orientations" and fit["weighting"] == "unweighted"
    )
    frequency_range = (
        f"{valid['frequency_lp_per_mm'].min():.2f}–"
        f"{valid['frequency_lp_per_mm'].max():.2f} lp/mm"
    )
    contrast_range = f"{valid['contrast'].min():.3f}–{valid['contrast'].max():.3f}"
    delta_sigma = (
        float(pooled_unweighted["sigma_f_lp_per_mm"])
        - float(pooled_weighted["sigma_f_lp_per_mm"])
    )
    delta_sigma_percent = 100.0 * delta_sigma / float(pooled_weighted["sigma_f_lp_per_mm"])
    delta_amplitude = (
        float(pooled_unweighted["amplitude_A"])
        - float(pooled_weighted["amplitude_A"])
    )

    lines = [
        f"# {PROJECT_TITLE}",
        "",
        "**Nikul Prajapati | Technische Hochschule Deggendorf (THD)**",
        "",
        "**Course:** Special Topics of Contactless Sensor Systems",
        "",
        "## Results",
        "",
        f"- Input image analyzed at native resolution: {EXPECTED_SHAPE[1]} × "
        f"{EXPECTED_SHAPE[0]} pixels.",
        f"- Pixel pitch: {PIXEL_PITCH_UM:.3f} µm = {PIXEL_PITCH_MM:.6f} mm; "
        f"Nyquist frequency: {NYQUIST_LP_PER_MM:.2f} lp/mm.",
        f"- Valid measured patterns: **{len(valid)} / {len(measurements)}**; "
        f"frequency range: **{frequency_range}**; CTF contrast range: **{contrast_range}**.",
        f"- Weighted pooled fit: A = {format_fit_value(pooled_weighted, 'amplitude_A', 'amplitude_A_std_error', '')}; "
        f"σf = {format_fit_value(pooled_weighted, 'sigma_f_lp_per_mm', 'sigma_f_std_error_lp_per_mm', 'lp/mm')}; "
        f"R² = {float(pooled_weighted['r_squared']):.3f}; RMSE = {float(pooled_weighted['rmse']):.4f}.",
        "",
        "σf is the **frequency-domain standard deviation**. A is the extrapolated zero-frequency "
        "contrast; DC was not directly measured.",
        "",
        "## Measurement and fitting method",
        "",
        "The original PNG is analyzed at native resolution. ROIs use editable half-open pixel "
        "coordinates in `mtf_analysis.py`; H04 and V04 were tightened after visual inspection of "
        "the original pixels. V04 was restricted to its vertical-bar patch, excluding the separate "
        "horizontal-bar patch immediately above it. For each ROI, parallel pixel lines are averaged "
        "to form a profile perpendicular to the stripes. Period is estimated from the profile's "
        "windowed Fourier peak, then converted using the measured pixel pitch.",
        "",
        "- `Period_mm = Period_pixels × 0.002143 mm/pixel`",
        "- `f (lp/mm) = 1 / Period_mm`",
        "- `C = (Ibright − Idark) / (Ibright + Idark)`",
        "- `C(f) = A exp[−f² / (2σf²)]`; normalized response `C(f)/A`.",
        "",
        "The measured stripes are bars, so their Michelson contrast is a **contrast transfer "
        "function (CTF)**. It is not a fully corrected sinusoidal MTF: no square-wave harmonic "
        "conversion was applied. The normalized Gaussian is an MTF estimate based on measured CTF.",
        "",
        "Weighted fitting minimizes residuals scaled by the contrast standard error from the "
        "parallel subprofiles (with a 0.005 floor); unweighted fitting gives each pattern equal "
        "weight. Both R² and RMSE are calculated on the unweighted contrast residuals.",
        "",
        "## Weighted versus unweighted Gaussian fits",
        "",
        "| Fit group | Weighting | N | A (extrapolated DC) | σf (lp/mm) | R² | RMSE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for fit in fits:
        r_squared = f"{float(fit['r_squared']):.3f}" if np.isfinite(fit["r_squared"]) else "n/a"
        rmse = f"{float(fit['rmse']):.4f}" if np.isfinite(fit["rmse"]) else "n/a"
        lines.append(
            f"| {fit['fit_group']} | {fit['weighting']} | {fit['n_measurements']} | "
            f"{format_fit_value(fit, 'amplitude_A', 'amplitude_A_std_error', '')} | "
            f"{format_fit_value(fit, 'sigma_f_lp_per_mm', 'sigma_f_std_error_lp_per_mm', '')} | "
            f"{r_squared} | {rmse} |"
        )
    lines.extend([
        "",
        f"For the pooled fits, unweighted minus weighted is Δσf = {delta_sigma:+.3f} lp/mm "
        f"({delta_sigma_percent:+.2f}%) and ΔA = {delta_amplitude:+.4f}. The weighted pooled "
        "result is used for the normalized plot. The weighted fit has R² = "
        f"{float(pooled_weighted['r_squared']):.3f}, while the unweighted fit has "
        f"R² = {float(pooled_unweighted['r_squared']):.3f}; the strong weighting dependence "
        "and weak weighted-fit R² mean that neither σf should be treated as a robust, unique "
        "system-resolution estimate.",
        "",
        "Formal parameter standard errors are conditional on this Gaussian model and the stated "
        "weights. They do not include systematic ROI-boundary uncertainty, uncertainty in pixel "
        "pitch, finite pattern geometry, or the CTF-to-MTF approximation. The subprofile spread "
        "captures within-ROI variability, not changes caused by moving an ROI boundary. Normalized "
        "points can exceed 1 because an individual measured CTF can exceed the fitted extrapolated "
        "amplitude A; this is fit/target variability, not evidence of physical MTF above unity.",
        "",
        "## Measured patterns",
        "",
        "| ROI | Pattern | Orientation | Period (px) | Period (mm) | Frequency (lp/mm) | Bright | Dark | Contrast ± SD |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in valid.itertuples(index=False):
        lines.append(
            f"| {row.roi_id} | {row.pattern} | {row.orientation} | "
            f"{row.period_px:.2f} | {row.period_mm:.5f} | {row.frequency_lp_per_mm:.2f} | "
            f"{row.bright_intensity_mean:.1f} | {row.dark_intensity_mean:.1f} | "
            f"{row.contrast:.3f} ± {row.contrast_sd:.3f} |"
        )

    lines.extend([
        "",
        "## Exclusions and limitations",
        "",
        "Printed markings, dot/ring patterns, borders, and nonperiodic background were not treated "
        "as stripe ROIs. Frequency is measured from the pixel data, not read from printed labels. "
        "Candidate ROIs with fewer than three cycles, weak periodicity, above-Nyquist frequency, "
        "or insufficient sampled stripe interiors are excluded:",
        "",
    ])
    for row in excluded.itertuples(index=False):
        lines.append(f"- **{row.roi_id} — {row.pattern}:** {row.exclusion_reason or 'Not usable.'}")
    lines.extend([
        "",
        "Finite ROI length, slight stripe tilt, image noise, and uncertain bar duty cycle also limit "
        "interpretation. `measurements.csv` includes all configured ROIs and exclusion reasons; "
        "`fit_results.csv` includes weighted and unweighted pooled and orientation-specific fits.",
        "",
        "## Figures",
        "",
        "- `intensity_profiles.png` — averaged grayscale profiles.",
        "- `representative_intensity_profiles.png` — selected profiles at low, medium, and high spatial frequencies.",
        "- `contrast_transfer_vs_spatial_frequency.png` — raw measured CTF.",
        "- `normalized_mtf_estimate_and_gaussian_fits.png` — normalized CTF points and Gaussian fits.",
        "- `../CT_Image_Spatial_Resolution_Analysis_Report.pdf` — image-free multipage report.",
        "",
        "The source CT image and its ROI overview are not included in this repository because "
        "redistribution permission could not be established. The PDF report contains plots, "
        "measurements, methods, and limitations, but does not embed the source image.",
        "",
        "## Development and AI Assistance",
        "",
        "I reviewed the project requirements, supplied the input image, defined the analysis "
        "requirements, and reviewed the outputs. GitHub Copilot assisted with the Python "
        "implementation, numerical calculations, plotting, and documentation. The project was "
        "tested using the original supplied image.",
        "",
    ])
    (RESULTS_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def save_report_pdf(
    image: np.ndarray,
    measurements: pd.DataFrame,
    fits: list[dict[str, object]],
    profiles_for_pdf: dict[str, tuple[np.ndarray, np.ndarray]],
) -> Path:
    pdf_path = PROJECT_DIR / "CT_Image_Spatial_Resolution_Analysis_Report.pdf"
    valid = measurements.loc[measurements["usable"]].copy()
    excluded = measurements.loc[~measurements["usable"]].copy()
    weighted_fits = [fit for fit in fits if fit["weighting"] == "weighted"]
    pooled_weighted = next(
        fit for fit in weighted_fits if fit["fit_group"] == "all orientations"
    )
    pooled_unweighted = next(
        fit for fit in fits
        if fit["fit_group"] == "all orientations" and fit["weighting"] == "unweighted"
    )
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.96, PROJECT_TITLE,
                 fontsize=17, fontweight="bold", color="#18324a", va="top")
        fig.text(
            0.08, 0.925,
            "Nikul Prajapati | Technische Hochschule Deggendorf (THD) | "
            "Special Topics of Contactless Sensor Systems",
            fontsize=8.3, color="#34495e", va="top",
        )
        summary_lines = [
            f"Input image dimensions: {image.shape[1]} × {image.shape[0]} px "
            "(source image not embedded)",
            f"Pixel pitch: {PIXEL_PITCH_UM:.3f} µm = {PIXEL_PITCH_MM:.6f} mm; "
            f"Nyquist: {NYQUIST_LP_PER_MM:.2f} lp/mm",
            f"Valid patterns: {len(valid)} / {len(measurements)} | "
            f"frequency: {valid.frequency_lp_per_mm.min():.2f}–"
            f"{valid.frequency_lp_per_mm.max():.2f} lp/mm | "
            f"CTF contrast: {valid.contrast.min():.3f}–{valid.contrast.max():.3f}",
            f"Weighted pooled fit: A = {float(pooled_weighted['amplitude_A']):.4f} ± "
            f"{float(pooled_weighted['amplitude_A_std_error']):.4f}; "
            f"σf = {float(pooled_weighted['sigma_f_lp_per_mm']):.2f} ± "
            f"{float(pooled_weighted['sigma_f_std_error_lp_per_mm']):.2f} lp/mm; "
            f"R² = {float(pooled_weighted['r_squared']):.3f}; "
            f"RMSE = {float(pooled_weighted['rmse']):.4f}",
        ]
        fig.text(0.08, 0.88, "Measured results", fontsize=12, fontweight="bold", color="#18324a")
        fig.text(
            0.08, 0.85, "\n".join(textwrap.fill(line, width=98) for line in summary_lines),
            fontsize=9.4, va="top", linespacing=1.45,
        )

        fit_rows = []
        for fit in fits:
            fit_rows.append([
                str(fit["fit_group"]).capitalize(),
                str(fit["weighting"]),
                str(fit["n_measurements"]),
                f"{float(fit['amplitude_A']):.4f} ± {float(fit['amplitude_A_std_error']):.4f}",
                f"{float(fit['sigma_f_lp_per_mm']):.2f} ± "
                f"{float(fit['sigma_f_std_error_lp_per_mm']):.2f}",
                f"{float(fit['r_squared']):.3f}",
                f"{float(fit['rmse']):.4f}",
            ])
        fig.text(0.08, 0.735, "Weighted versus unweighted Gaussian fits",
                 fontsize=12, fontweight="bold", color="#18324a")
        table_ax = fig.add_axes([0.06, 0.49, 0.88, 0.22])
        table_ax.axis("off")
        fit_table = table_ax.table(
            cellText=fit_rows,
            colLabels=["Fit group", "Weighting", "N", "A ± SE", "σf ± SE (lp/mm)", "R²", "RMSE"],
            colWidths=[0.19, 0.14, 0.08, 0.20, 0.18, 0.10, 0.11],
            cellLoc="center",
            colLoc="center",
            bbox=[0.0, 0.0, 1.0, 1.0],
        )
        fit_table.auto_set_font_size(False)
        fit_table.set_fontsize(7.5)
        for (row_index, _), cell in fit_table.get_celld().items():
            if row_index == 0:
                cell.set_facecolor("#dce7ef")
                cell.set_text_props(weight="bold", color="#18324a")
            elif row_index % 2:
                cell.set_facecolor("#f3f6f8")

        delta_sigma = (
            float(pooled_unweighted["sigma_f_lp_per_mm"])
            - float(pooled_weighted["sigma_f_lp_per_mm"])
        )
        delta_percent = 100.0 * delta_sigma / float(pooled_weighted["sigma_f_lp_per_mm"])
        delta_a = float(pooled_unweighted["amplitude_A"]) - float(pooled_weighted["amplitude_A"])
        interpretation = (
            f"Pooled unweighted − weighted: Δσf = {delta_sigma:+.3f} lp/mm "
            f"({delta_percent:+.2f}%), ΔA = {delta_a:+.4f}.\n\n"
            "Period_mm = Period_px × 0.002143 mm/px; f = 1/Period_mm.\n"
            "C = (Ibright − Idark)/(Ibright + Idark); C(f) = A exp[−f²/(2σf²)]; "
            "normalized response = C(f)/A.\n\n"
            "Measured bar contrast is a CTF. No square-wave harmonic conversion was applied; "
            "the normalized curve is an MTF estimate, not a fully corrected sinusoidal MTF. "
            "A is extrapolated to DC; DC was not measured."
        )
        interpretation += (
            f"\n\nFit quality is weighting-sensitive: pooled R² = "
            f"{float(pooled_weighted['r_squared']):.3f} weighted and "
            f"{float(pooled_unweighted['r_squared']):.3f} unweighted. "
            "The weak weighted-fit R² and large σf difference make these descriptive fits, "
            "not a robust unique resolution estimate."
        )
        fig.text(0.08, 0.455, "Interpretation", fontsize=12, fontweight="bold", color="#18324a")
        fig.text(
            0.08, 0.425, wrap_pdf_text(interpretation, width=98),
            fontsize=9.0, va="top", linespacing=1.25,
        )
        limitations = (
            "Why normalized points can exceed 1: individual measured CTF values may exceed the fitted "
            "zero-frequency amplitude A because of measurement, target, and model variability; "
            "this is not physical MTF above unity.\n"
            "Uncertainty: curve-fit standard errors are conditional on the Gaussian model and stated "
            "weights. They omit systematic ROI-boundary and pixel-pitch uncertainty, finite geometry, "
            "and CTF-to-MTF bias. Subprofile spread measures within-ROI variability, not ROI-placement "
            "sensitivity.\n"
            f"Excluded candidates: {', '.join(excluded['roi_id'].astype(str)) or 'none'}."
        )
        fig.text(0.08, 0.235, "Limitations", fontsize=12, fontweight="bold", color="#18324a")
        fig.text(
            0.08, 0.205, wrap_pdf_text(limitations, width=98),
            fontsize=8.8, va="top", linespacing=1.2,
        )
        fig.text(
            0.08, 0.065,
            wrap_pdf_text(
                "Development and AI assistance: I reviewed the project requirements, supplied the "
                "input image, defined the analysis requirements, and reviewed the outputs. GitHub "
                "Copilot assisted with Python implementation, calculations, plots, and documentation.",
                width=100,
            ),
            fontsize=7.6, color="#5b6570", va="bottom",
        )
        pdf.savefig(fig)
        plt.close(fig)

        valid_rows = list(valid.itertuples(index=False))
        for page_number, start in enumerate(range(0, len(valid_rows), 10), start=1):
            page_rows = valid_rows[start:start + 10]
            fig, axes = plt.subplots(5, 2, figsize=(11.69, 8.27), squeeze=False)
            fig.suptitle(
                f"Measured grayscale intensity profiles ({start + 1}–{start + len(page_rows)} of {len(valid_rows)})",
                fontsize=13, fontweight="bold", color="#18324a", y=0.95,
            )
            fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.88,
                                hspace=0.58, wspace=0.22)
            fig.text(
                0.5, 0.91,
                "Solid: averaged profile   |   dashed: bright mean   |   dotted: dark mean",
                ha="center", fontsize=7.5, color="#34495e",
            )
            for ax, row in zip(axes.flat, page_rows):
                profile, coordinate = profiles_for_pdf[row.roi_id]
                ax.plot(coordinate, profile, color="#18324a", linewidth=1.0)
                ax.axhline(row.bright_intensity_mean, color="#d97706", linestyle="--", linewidth=0.8)
                ax.axhline(row.dark_intensity_mean, color="#176b87", linestyle=":", linewidth=0.9)
                ax.set_title(f"{row.roi_id} — {row.orientation} stripes", fontsize=8.5)
                ax.set_xlabel(f"{row.variation_axis} position (px)", fontsize=7.5)
                ax.set_ylabel("Gray (0–255)", fontsize=7.5)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.2)
            for ax in axes.flat[len(page_rows):]:
                ax.remove()
            pdf.savefig(fig)
            plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.suptitle("Measured CTF and normalized Gaussian MTF estimate",
                     fontsize=13, fontweight="bold", color="#18324a", y=0.96)
        fig.subplots_adjust(left=0.09, right=0.97, bottom=0.12, top=0.88, wspace=0.30)
        orientation_colors = {"vertical": "#176b87", "horizontal": "#d97706"}
        raw_ax, mtf_ax = axes
        for orientation, group in valid.groupby("orientation"):
            raw_ax.errorbar(
                group.frequency_lp_per_mm, group.contrast, yerr=group.contrast_sd,
                fmt="o", capsize=2, markersize=4, color=orientation_colors[orientation],
                label=f"{orientation.capitalize()} stripes",
            )
        raw_ax.set_xlabel("Spatial frequency (lp/mm)")
        raw_ax.set_ylabel("Michelson contrast (CTF)")
        raw_ax.set_title("Raw measured bar-pattern contrast")
        raw_ax.set_xlim(left=0)
        raw_ax.set_ylim(bottom=0)
        raw_ax.grid(True, alpha=0.25)
        raw_ax.legend(fontsize=8)

        amplitude = float(pooled_weighted["amplitude_A"])
        for orientation, group in valid.groupby("orientation"):
            mtf_ax.errorbar(
                group.frequency_lp_per_mm, group.contrast / amplitude,
                yerr=group.contrast_sd / amplitude, fmt="o", capsize=2,
                markersize=4, color=orientation_colors[orientation],
                label=f"{orientation.capitalize()} CTF / A",
            )
        fit_frequencies = np.linspace(0.0, float(valid.frequency_lp_per_mm.max()), 400)
        for fit in weighted_fits:
            sigma_f = float(fit["sigma_f_lp_per_mm"])
            if fit["fit_group"] == "all orientations":
                color, linestyle, label = "#18324a", "-", "Pooled weighted Gaussian"
            else:
                color = orientation_colors[str(fit["fit_group"])]
                linestyle, label = "--", f"{str(fit['fit_group']).capitalize()} weighted fit"
            mtf_ax.plot(
                fit_frequencies,
                np.exp(-(fit_frequencies**2) / (2.0 * sigma_f**2)),
                color=color, linestyle=linestyle, linewidth=1.5, label=label,
            )
        unweighted_sigma = float(pooled_unweighted["sigma_f_lp_per_mm"])
        mtf_ax.plot(
            fit_frequencies,
            np.exp(-(fit_frequencies**2) / (2.0 * unweighted_sigma**2)),
            color="#7353a6", linestyle=":", linewidth=1.5,
            label="Pooled unweighted Gaussian",
        )
        mtf_ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
        mtf_ax.set_xlabel("Spatial frequency (lp/mm)")
        mtf_ax.set_ylabel("Normalized response, C(f) / A")
        mtf_ax.set_title("Estimated MTF (CTF-based; not harmonic-corrected)")
        mtf_ax.set_xlim(left=0)
        mtf_ax.set_ylim(bottom=0)
        mtf_ax.grid(True, alpha=0.25)
        mtf_ax.legend(fontsize=7.5)
        pdf.savefig(fig)
        plt.close(fig)

        for page_index, start in enumerate(range(0, len(valid_rows), 15), start=1):
            page_rows = valid_rows[start:start + 15]
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis("off")
            fig.suptitle(
                f"Measured pattern values ({start + 1}–{start + len(page_rows)} of {len(valid_rows)})",
                fontsize=13, fontweight="bold", color="#18324a", y=0.96,
            )
            table_rows = []
            for row in page_rows:
                pattern = row.pattern.replace(" bars", "").replace(", ", " — ")
                table_rows.append([
                    row.roi_id,
                    row.orientation,
                    pattern,
                    f"{row.period_px:.2f}",
                    f"{row.period_mm:.5f}",
                    f"{row.frequency_lp_per_mm:.2f}",
                    f"{row.bright_intensity_mean:.1f} / {row.dark_intensity_mean:.1f}",
                    f"{row.contrast:.3f} ± {row.contrast_sd:.3f}",
                ])
            table = ax.table(
                cellText=table_rows,
                colLabels=[
                    "ROI", "Orientation", "Pattern", "Period (px)", "Period (mm)",
                    "Frequency (lp/mm)", "Ibright / Idark", "Michelson C ± SD",
                ],
                colWidths=[0.06, 0.10, 0.26, 0.10, 0.10, 0.13, 0.13, 0.12],
                cellLoc="center",
                colLoc="center",
                bbox=[0.02, 0.06, 0.96, 0.84],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7.8)
            for (row_index, _), cell in table.get_celld().items():
                if row_index == 0:
                    cell.set_facecolor("#dce7ef")
                    cell.set_text_props(weight="bold", color="#18324a")
                elif row_index % 2:
                    cell.set_facecolor("#f3f6f8")
            if start + len(page_rows) == len(valid_rows) and not excluded.empty:
                excluded_text = "; ".join(
                    f"{row.roi_id}: {row.exclusion_reason}" for row in excluded.itertuples(index=False)
                )
                fig.text(0.04, 0.035, f"Excluded: {excluded_text}", fontsize=7.3, color="#5b6570")
            pdf.savefig(fig)
            plt.close(fig)

    return pdf_path


def run_analysis(show: bool = False) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    image = load_image()
    rois = validate_rois(image)

    records: list[dict[str, object]] = []
    profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for roi in rois:
        record, profile, coordinate = measure_roi(image, roi)
        records.append(record)
        profiles[roi.roi_id] = (profile, coordinate)
    measurements = pd.DataFrame.from_records(records)
    measurements.to_csv(RESULTS_DIR / "measurements.csv", index=False, float_format="%.8g")
    valid = measurements.loc[measurements["usable"]].copy()
    if valid.empty:
        raise RuntimeError(
            "No periodic stripe ROIs passed quality checks. Review ROI_CONFIG; "
            "no fabricated measurements will be used."
        )

    fit_groups = [("all orientations", valid)]
    fit_groups.extend(
        (orientation, valid.loc[valid["orientation"] == orientation])
        for orientation in ("vertical", "horizontal")
    )
    fits = [
        fit_gaussian(group, name, weighted)
        for name, group in fit_groups
        for weighted in (True, False)
    ]
    fit_results = pd.DataFrame.from_records(fits)
    fit_results.to_csv(RESULTS_DIR / "fit_results.csv", index=False, float_format="%.8g")

    save_roi_overview(image, rois, measurements)
    save_intensity_profiles(profiles, measurements)
    save_representative_profiles(profiles, measurements)
    save_contrast_plot(measurements)
    save_mtf_plot(measurements, fits)
    write_report(measurements, fits)
    pdf_path = save_report_pdf(image, measurements, fits, profiles)

    pooled = next(
        fit for fit in fits
        if fit["fit_group"] == "all orientations" and fit["weighting"] == "weighted"
    )
    unweighted = next(
        fit for fit in fits
        if fit["fit_group"] == "all orientations" and fit["weighting"] == "unweighted"
    )
    print(f"Measured patterns: {len(valid)} / {len(rois)} configured ROIs")
    print(
        "Frequency range: "
        f"{valid['frequency_lp_per_mm'].min():.2f}–"
        f"{valid['frequency_lp_per_mm'].max():.2f} lp/mm"
    )
    print(
        "Contrast range: "
        f"{valid['contrast'].min():.3f}–{valid['contrast'].max():.3f}"
    )
    if np.isfinite(pooled["sigma_f_lp_per_mm"]):
        print(
            f"Gaussian σf: {pooled['sigma_f_lp_per_mm']:.3g} lp/mm "
            f"(weighted; A={pooled['amplitude_A']:.3g}, R²={pooled['r_squared']:.3f})"
        )
        print(
            f"Unweighted Gaussian σf: {unweighted['sigma_f_lp_per_mm']:.3g} lp/mm "
            f"(Δσf={float(unweighted['sigma_f_lp_per_mm']) - float(pooled['sigma_f_lp_per_mm']):+.3g} lp/mm)"
        )
    else:
        print(f"Gaussian fit: {pooled['fit_status']}")
    print(f"Nyquist: {NYQUIST_LP_PER_MM:.2f} lp/mm")
    print(f"Generated files: {RESULTS_DIR}")
    print(f"PDF report: {pdf_path}")

    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="Display plots interactively after saving.")
    args = parser.parse_args()
    try:
        run_analysis(show=args.show)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        print(f"MTF analysis failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
