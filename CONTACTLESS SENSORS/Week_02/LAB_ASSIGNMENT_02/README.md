# Week 02 - Fourier Transform Pairs

This assignment compares the Fourier transform of a unit-area rectangular pulse with its theoretical sinc transform, then studies the discrete harmonics of a periodic rectangle train. The supplied cosine-sign formula is used as the fixed 50% duty-cycle example. A separate, explicitly modified periodic rectangle is used to vary pulse width.

## Files

- `Nikul_Week_02_Fourier_Transform_Pairs.ipynb` - explanations, plots, FFT calculations, and numerical checks.
- `Nikul_Week_02_Viva_Preparation.md` - 20 viva questions and concise answers.
- `requirements.txt` - the two Python packages used by the notebook.

## Requirements

- Python 3.14 or newer
- NumPy
- Matplotlib
- Jupyter Notebook support

## Installation

From this directory, create and activate a virtual environment using Python 3.14, then install the listed packages:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install Jupyter Notebook support in the same environment if it is not already available:

```bash
python -m pip install notebook ipykernel
```

## Run

Open `Nikul_Week_02_Fourier_Transform_Pairs.ipynb` in Jupyter or VS Code, select the Python 3.14 environment, and run all cells from top to bottom. The notebook includes assertions for pulse height and area, the zero-frequency value, frequency-bin alignment, and agreement between the numerical FFT and theoretical results.
