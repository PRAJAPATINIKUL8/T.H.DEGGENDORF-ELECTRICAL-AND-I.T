"""Create a portrait LinkedIn carousel from the project's measured result plots."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
OUTPUT_PATH = PROJECT_DIR / "CT_Image_Spatial_Resolution_Analysis_LinkedIn_Carousel.pdf"
PROJECT_TITLE = "CT Image Spatial Resolution Analysis Using Python"
PAGE_SIZE_INCHES = (8.0, 10.0)


def add_page_header(fig: plt.Figure, page: int, section: str) -> None:
    fig.text(
        0.065, 0.969, "CT IMAGE SPATIAL RESOLUTION",
        fontsize=8.5, fontweight="bold", color="#176b87", va="top",
    )
    fig.text(
        0.935, 0.969, f"{page:02d} / 05",
        fontsize=8, color="#5b6570", ha="right", va="top",
    )
    fig.add_artist(Line2D(
        [0.065, 0.935], [0.945, 0.945], transform=fig.transFigure,
        color="#dce7ef", linewidth=1.2,
    ))
    fig.text(
        0.065, 0.917, section.upper(),
        fontsize=7.5, color="#5b6570", fontweight="bold", va="top",
    )


def add_title(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.text(
        0.065, 0.875, title, fontsize=22, fontweight="bold",
        color="#18324a", va="top", wrap=True,
    )
    if subtitle:
        fig.text(
            0.067, 0.805, subtitle, fontsize=10.2,
            color="#34495e", va="top", wrap=True,
        )


def add_plot_image(
    fig: plt.Figure,
    path: Path,
    box: tuple[float, float, float, float],
) -> None:
    with Image.open(path) as source:
        source.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
        image = np.asarray(source.convert("RGB"))

    left, bottom, box_width, box_height = box
    image_aspect = image.shape[1] / image.shape[0]
    box_aspect = box_width * PAGE_SIZE_INCHES[0] / (box_height * PAGE_SIZE_INCHES[1])
    if image_aspect > box_aspect:
        width = box_width
        height = width * PAGE_SIZE_INCHES[0] / (image_aspect * PAGE_SIZE_INCHES[1])
    else:
        height = box_height
        width = height * PAGE_SIZE_INCHES[1] * image_aspect / PAGE_SIZE_INCHES[0]
    axes = fig.add_axes((
        left + (box_width - width) / 2,
        bottom + (box_height - height) / 2,
        width,
        height,
    ))
    axes.imshow(image)
    axes.set_axis_off()


def add_footer(fig: plt.Figure, text: str) -> None:
    fig.text(
        0.065, 0.055, text, fontsize=8.1, color="#5b6570",
        va="bottom", wrap=True,
    )


def generate_carousel() -> Path:
    required_figures = {
        "profiles": RESULTS_DIR / "representative_intensity_profiles.png",
        "ctf": RESULTS_DIR / "contrast_transfer_vs_spatial_frequency.png",
        "mtf": RESULTS_DIR / "normalized_mtf_estimate_and_gaussian_fits.png",
    }
    missing = [path for path in required_figures.values() if not path.is_file()]
    if missing:
        names = ", ".join(str(path.relative_to(PROJECT_DIR)) for path in missing)
        raise FileNotFoundError(
            f"Missing generated plot(s): {names}. Run mtf_analysis.py first."
        )

    measurements = pd.read_csv(RESULTS_DIR / "measurements.csv")
    fits = pd.read_csv(RESULTS_DIR / "fit_results.csv")
    usable = measurements.loc[measurements["usable"].astype(str).str.lower() == "true"]
    pooled = fits.loc[fits["fit_group"] == "all orientations"].set_index("weighting")
    if usable.empty or not {"weighted", "unweighted"}.issubset(pooled.index):
        raise ValueError("The result CSVs do not contain the expected usable patterns and pooled fits.")

    weighted = pooled.loc["weighted"]
    unweighted = pooled.loc["unweighted"]
    frequency_min = float(usable["frequency_lp_per_mm"].min())
    frequency_max = float(usable["frequency_lp_per_mm"].max())
    contrast_min = float(usable["contrast"].min())
    contrast_max = float(usable["contrast"].max())
    sigma_delta = float(unweighted["sigma_f_lp_per_mm"] - weighted["sigma_f_lp_per_mm"])
    sigma_delta_percent = 100.0 * sigma_delta / float(weighted["sigma_f_lp_per_mm"])

    pdf_buffer = BytesIO()
    with PdfPages(
        pdf_buffer,
        metadata={
            "Title": PROJECT_TITLE + " — LinkedIn Carousel",
            "Author": "Nikul Prajapati",
            "Subject": "CT bar-pattern contrast and Gaussian fitting",
        },
    ) as pdf:
        fig = plt.figure(figsize=PAGE_SIZE_INCHES, facecolor="#f4f7fa")
        add_page_header(fig, 1, "Project overview")
        fig.text(
            0.065, 0.87, PROJECT_TITLE,
            fontsize=25, fontweight="bold", color="#18324a",
            va="top", wrap=True,
        )
        fig.text(
            0.067, 0.745,
            "Measuring spatial response from periodic CT bar patterns",
            fontsize=12, color="#34495e", va="top",
        )
        fig.text(
            0.067, 0.694,
            "Nikul Prajapati  |  Technische Hochschule Deggendorf (THD)  |  "
            "Special Topics of Contactless Sensor Systems",
            fontsize=7.6, color="#5b6570", va="top",
        )
        fig.text(
            0.08, 0.55, f"{len(usable)}",
            fontsize=44, color="#176b87", fontweight="bold",
        )
        fig.text(
            0.31, 0.59, "usable measured patterns",
            fontsize=15, color="#18324a", fontweight="bold",
        )
        fig.text(
            0.08, 0.43,
            f"{frequency_min:.2f}–{frequency_max:.2f} lp/mm",
            fontsize=19, color="#18324a", fontweight="bold",
        )
        fig.text(
            0.08, 0.375, "measured spatial-frequency range",
            fontsize=10, color="#5b6570",
        )
        fig.text(
            0.08, 0.29, "2.143 µm pixel pitch",
            fontsize=13, color="#176b87", fontweight="bold",
        )
        add_footer(fig, "Bar contrast measures CTF; the normalized Gaussian is an MTF estimate.")
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        fig = plt.figure(figsize=PAGE_SIZE_INCHES, facecolor="#f4f7fa")
        add_page_header(fig, 2, "From image to measurements")
        add_title(
            fig,
            "Grayscale profiles reveal the stripe periods",
            "Parallel scan lines are averaged to reduce local noise while preserving the measured bar modulation.",
        )
        add_plot_image(fig, required_figures["profiles"], (0.08, 0.115, 0.84, 0.66))
        add_footer(fig, "Profiles and ROI coordinates are derived from the supplied image; the source image is not included.")
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        fig = plt.figure(figsize=PAGE_SIZE_INCHES, facecolor="#f4f7fa")
        add_page_header(fig, 3, "Measured contrast")
        add_title(
            fig,
            "Raw bar contrast falls with spatial frequency",
            f"Michelson contrast spans {contrast_min:.3f}–{contrast_max:.3f}; frequency spans "
            f"{frequency_min:.2f}–{frequency_max:.2f} lp/mm.",
        )
        add_plot_image(fig, required_figures["ctf"], (0.07, 0.245, 0.86, 0.49))
        fig.text(
            0.08, 0.165,
            "C = (Ibright − Idark) / (Ibright + Idark)",
            fontsize=12, color="#18324a", fontweight="bold",
        )
        add_footer(fig, "Bars measure a contrast transfer function (CTF), not a corrected sinusoidal MTF.")
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        fig = plt.figure(figsize=PAGE_SIZE_INCHES, facecolor="#f4f7fa")
        add_page_header(fig, 4, "Gaussian response estimate")
        add_title(
            fig,
            "Normalize to an extrapolated DC reference",
            "C(f) = A exp[−f² / (2σf²)]  ·  MTF estimate = C(f) / A",
        )
        add_plot_image(fig, required_figures["mtf"], (0.07, 0.27, 0.86, 0.47))
        fig.text(
            0.08, 0.185,
            f"Weighted σf = {float(weighted['sigma_f_lp_per_mm']):.2f} lp/mm"
            f"  |  Unweighted σf = {float(unweighted['sigma_f_lp_per_mm']):.2f} lp/mm",
            fontsize=9.6, color="#18324a", fontweight="bold",
        )
        add_footer(
            fig,
            f"DC is extrapolated, not measured. Normalized points may exceed 1 when a measured CTF "
            f"exceeds fitted A. Fit width differs by {sigma_delta:+.2f} lp/mm ({sigma_delta_percent:+.1f}%).",
        )
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        fig = plt.figure(figsize=PAGE_SIZE_INCHES, facecolor="#f4f7fa")
        add_page_header(fig, 5, "Results and limitations")
        add_title(fig, "The fit is informative, not definitive")

        table_ax = fig.add_axes([0.075, 0.57, 0.85, 0.17])
        table_ax.axis("off")
        table = table_ax.table(
            cellText=[
                [
                    "Weighted",
                    f"{float(weighted['amplitude_A']):.4f} ± {float(weighted['amplitude_A_std_error']):.4f}",
                    f"{float(weighted['sigma_f_lp_per_mm']):.2f} ± "
                    f"{float(weighted['sigma_f_std_error_lp_per_mm']):.2f}",
                    f"{float(weighted['r_squared']):.3f}",
                ],
                [
                    "Unweighted",
                    f"{float(unweighted['amplitude_A']):.4f} ± {float(unweighted['amplitude_A_std_error']):.4f}",
                    f"{float(unweighted['sigma_f_lp_per_mm']):.2f} ± "
                    f"{float(unweighted['sigma_f_std_error_lp_per_mm']):.2f}",
                    f"{float(unweighted['r_squared']):.3f}",
                ],
            ],
            colLabels=["Fit", "A (extrapolated DC)", "σf (lp/mm)", "R²"],
            cellLoc="center",
            colLoc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        for (row_index, _), cell in table.get_celld().items():
            if row_index == 0:
                cell.set_facecolor("#dce7ef")
                cell.set_text_props(weight="bold", color="#18324a")
            elif row_index % 2:
                cell.set_facecolor("#ffffff")

        fig.text(
            0.08, 0.50,
            f"29 usable ROIs from 31 candidates  ·  {frequency_min:.2f}–{frequency_max:.2f} lp/mm  ·  "
            f"weighted/unweighted σf difference: {sigma_delta:+.2f} lp/mm",
            fontsize=9.2, color="#176b87", fontweight="bold", wrap=True,
        )
        fig.text(
            0.08, 0.425,
            "Interpretation",
            fontsize=12, fontweight="bold", color="#18324a",
        )
        fig.text(
            0.08, 0.39,
            "The pooled fits are weighting-sensitive (R² = "
            f"{float(weighted['r_squared']):.3f} weighted; {float(unweighted['r_squared']):.3f} "
            "unweighted). Their widths are descriptive frequency-domain Gaussian standard deviations, "
            "not a unique scanner-resolution measurement. Fit standard errors do not include systematic "
            "ROI placement, pixel-pitch, finite-pattern, or CTF-to-MTF uncertainty.",
            fontsize=9.3, color="#34495e", va="top", linespacing=1.35, wrap=True,
        )
        fig.text(
            0.08, 0.225,
            "Development and AI assistance",
            fontsize=11, fontweight="bold", color="#18324a",
        )
        fig.text(
            0.08, 0.19,
            "I supplied the input image, defined the analysis requirements, and reviewed the outputs. "
            "GitHub Copilot assisted with implementation, calculations, plots, and documentation. "
            "The analysis was tested using the original supplied image.",
            fontsize=8.7, color="#34495e", va="top", linespacing=1.3, wrap=True,
        )
        add_footer(fig, "Pixel pitch: 2.143 µm. Source CT image and university course materials are not distributed.")
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

    OUTPUT_PATH.write_bytes(pdf_buffer.getvalue())
    pdf_buffer.close()
    return OUTPUT_PATH


if __name__ == "__main__":
    print(f"Created LinkedIn carousel: {generate_carousel()}")
