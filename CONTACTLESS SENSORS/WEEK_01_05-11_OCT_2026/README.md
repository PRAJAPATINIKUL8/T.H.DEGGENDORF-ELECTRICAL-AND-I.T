# Week 01 – Statistical Analysis

- **Student:** Nikul
- **University:** TH Deggendorf
- **Course:** Special Topics in Contactless Sensor Systems
- **Semester:** Winter Semester 2026/27
- **Week:** 05–11 October 2026
- **Assignment:** Home Exercise 01 – Statistical Analysis

## Project

The Python program reads the 100 original measurements from
`Nikul_noise_data.csv`. It calculates the count, mean, median, sample standard
deviation, minimum, and maximum, then saves the results and two plots in this
folder. The CSV contains only its column header and measurement data; student
details are not added to its rows.

The measurements plot shows each value against its slice number. The histogram
uses 10 bins and marks the mean. Both plots identify the student, university,
course, semester, week, and assignment.

## Requirements and run

The project uses Python, NumPy, and Matplotlib. The dependency versions are
listed in `requirements.txt`.

From this folder, create and activate a virtual environment, install the
requirements, and run the analysis:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python nikul_analyze_noise.py
```

The script locates the CSV beside itself and writes `Nikul_results.txt`,
`Nikul_measurements.png`, and `Nikul_histogram.png` to this folder.

## Results

The supplied data produces 100 measurements, a mean of 45006.45, a median of
45006.50, a sample standard deviation of 164.29761, a minimum of 44644, and a
maximum of 45338.
