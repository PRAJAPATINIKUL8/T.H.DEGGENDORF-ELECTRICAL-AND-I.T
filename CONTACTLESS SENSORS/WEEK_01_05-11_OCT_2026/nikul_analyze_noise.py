# ============================================================
# Student Name : Nikul
# University   : TH Deggendorf
# Course       : Special Topics in Contactless Sensor Systems
# Semester     : Winter Semester 2026/27
# Assignment   : Home Exercise 01 – Statistical Analysis
# Week         : 05–11 October 2026
# File         : nikul_analyze_noise.py
#
# Description:
# Analyze the supplied noise measurements and save descriptive
# statistics and plots in this project directory.
# ============================================================

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


STUDENT = "Nikul"
UNIVERSITY = "TH Deggendorf"
COURSE = "Special Topics in Contactless Sensor Systems"
SEMESTER = "Winter Semester 2026/27"
ASSIGNMENT = "Home Exercise 01 – Statistical Analysis"
WEEK = "05–11 October 2026"
DATASET_NAME = "Nikul_noise_data.csv"


def load_measurements(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the Slice and Mean columns from the supplied CSV."""
    data = np.genfromtxt(
        csv_path,
        delimiter=",",
        names=True,
        dtype=float,
        encoding="utf-8",
        ndmin=1,
    )
    columns = data.dtype.names or ()
    required_columns = {"Slice", "Mean"}
    missing_columns = required_columns.difference(columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{csv_path.name} is missing required column(s): {missing}")

    slices = np.asarray(data["Slice"], dtype=float)
    measurements = np.asarray(data["Mean"], dtype=float)
    if slices.size == 0 or measurements.size == 0:
        raise ValueError(f"{csv_path.name} does not contain any measurements.")
    if slices.size != measurements.size:
        raise ValueError("Slice and Mean columns must contain the same number of values.")
    if not np.all(np.isfinite(slices)) or not np.all(np.isfinite(measurements)):
        raise ValueError("Slice and Mean columns must contain only finite numeric values.")
    if measurements.size < 2:
        raise ValueError("At least two measurements are required for sample standard deviation.")

    return slices, measurements


def create_results_text(measurements: np.ndarray) -> str:
    """Calculate the required statistics and format the report."""
    count = measurements.size
    mean = np.mean(measurements)
    median = np.median(measurements)
    sample_std = np.std(measurements, ddof=1)
    minimum = np.min(measurements)
    maximum = np.max(measurements)

    interpretation = (
        f"The mean and median differ by {abs(mean - median):.2f}, so the two central "
        f"measures are nearly identical. The sample standard deviation is "
        f"{sample_std:.2f}. Measurements range from {minimum:g} to {maximum:g}, "
        f"a total range of {maximum - minimum:g}."
    )

    return "\n".join(
        [
            "Week 01 Python Statistical Analysis",
            f"Student: {STUDENT}",
            f"University: {UNIVERSITY}",
            f"Course: {COURSE}",
            f"Semester: {SEMESTER}",
            f"Assignment: {ASSIGNMENT}",
            f"Week: {WEEK}",
            f"Dataset: {DATASET_NAME}",
            "",
            f"Number of measurements: {count}",
            f"Mean: {mean:.2f}",
            f"Median: {median:.2f}",
            f"Sample standard deviation: {sample_std:.5f}",
            f"Minimum: {minimum:g}",
            f"Maximum: {maximum:g}",
            "",
            f"Interpretation: {interpretation}",
        ]
    )


def save_measurements_plot(
    slices: np.ndarray, measurements: np.ndarray, output_path: Path
) -> None:
    """Plot the measurements against their slice numbers."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(slices, measurements, marker="o", markersize=3, linewidth=1)
    ax.set_xlabel("Slice")
    ax.set_ylabel("Mean measurement")
    ax.set_title(f"{ASSIGNMENT}: Noise Measurements by Slice")
    ax.grid(True, alpha=0.4)
    fig.text(
        0.5,
        0.09,
        f"Student: {STUDENT} | University: {UNIVERSITY}",
        ha="center",
    )
    fig.text(0.5, 0.06, f"{COURSE} | {SEMESTER}", ha="center")
    fig.text(0.5, 0.03, f"Week 01: {WEEK} | {ASSIGNMENT}", ha="center")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_histogram(measurements: np.ndarray, output_path: Path) -> None:
    """Save a 10-bin histogram and indicate the mean measurement."""
    mean = np.mean(measurements)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(measurements, bins=10, edgecolor="black")
    ax.axvline(mean, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean:.2f}")
    ax.set_xlabel("Mean measurement")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{ASSIGNMENT}: Distribution of Noise Measurements")
    ax.legend()
    fig.text(
        0.5,
        0.09,
        f"Student: {STUDENT} | University: {UNIVERSITY}",
        ha="center",
    )
    fig.text(0.5, 0.06, f"{COURSE} | {SEMESTER}", ha="center")
    fig.text(0.5, 0.03, f"Week 01: {WEEK} | {ASSIGNMENT}", ha="center")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    csv_path = project_dir / DATASET_NAME
    slices, measurements = load_measurements(csv_path)

    results_text = create_results_text(measurements)
    results_path = project_dir / "Nikul_results.txt"
    results_path.write_text(results_text + "\n", encoding="utf-8")

    save_measurements_plot(
        slices,
        measurements,
        project_dir / "Nikul_measurements.png",
    )
    save_histogram(measurements, project_dir / "Nikul_histogram.png")

    print("=" * 58)
    print(ASSIGNMENT)
    print(f"Student: {STUDENT}")
    print(f"University: {UNIVERSITY}")
    print(f"Course: {COURSE}")
    print("=" * 58)
    print(results_text)


if __name__ == "__main__":
    main()
