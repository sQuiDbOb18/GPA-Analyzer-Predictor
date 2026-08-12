# GPA Analyzer & Predictor

A Jupyter Notebook app for calculating GPA, CGPA, and simple academic projections using the Nigerian university 5-point grading scale.

## Features

- Loads and validates student result data from CSV.
- Calculates per-semester GPA.
- Calculates cumulative CGPA after each semester.
- Plots GPA and CGPA trends on one chart.
- Projects CGPA after a hypothetical future semester.
- Calculates the average grade point needed to reach a target CGPA.
- Optionally projects the next CGPA using a simple regression trend predictor.

## Presentation

[View the presentation slide](./gpa_analyzer_presentation.pdf)

[View the notebook output (HTML)](./gpa_analyzer.html)

## Tech Stack

- Python 3.11+
- Pandas
- Matplotlib
- Jupyter Notebook
- Pytest
- Scikit-learn, optional for the regression trend predictor

## Project Structure

```text
gpa_analyzer/
  sample_results.csv          # Sample result data used when results.csv is not present
  sample_results_broken.csv   # Optional malformed sample file for validation checks
  gpa_analyzer.ipynb          # Main notebook interface, run this from top to bottom
  gpa_core.py                 # Pure calculation and validation functions
  test_gpa_core.py            # Pytest unit tests for the core logic
  requirements.txt            # Required Python packages, with scikit-learn marked optional
README.md                     # Submission documentation
```

## Setup Instructions

Clone the project and enter the folder:

```bash
git clone <repo-url>
cd "GPA Analyzer & Predictor"
```

Install the required dependencies:

```bash
python3.11 -m pip install -r gpa_analyzer/requirements.txt
```

Open the notebook:

```bash
cd gpa_analyzer
jupyter notebook gpa_analyzer.ipynb
```

If your system uses `python` instead of `python3.11`, use:

```bash
python -m pip install -r gpa_analyzer/requirements.txt
```

## How to Run

The notebook uses `results.csv` if that file exists in the `gpa_analyzer` folder. If `results.csv` is not present, it falls back to `sample_results.csv` so the app can run immediately.

Open `gpa_analyzer.ipynb`, then choose Restart Kernel and Run All. The notebook will print the validation summary, show the GPA/CGPA table, draw the trend chart, run a sample scenario projection, print target-CGPA advice, and run or skip the optional regression trend predictor.

## How to Test

From the `gpa_analyzer` folder, run:

```bash
pytest -v test_gpa_core.py
```

Expected result:

```text
8 passed
```

The 8 tests check:

- `test_gpa_single_semester`: verifies one semester GPA against a hand calculation.
- `test_cgpa_across_semesters`: verifies cumulative CGPA across multiple semesters.
- `test_invalid_grade_dropped`: verifies an invalid grade is excluded and counted.
- `test_zero_units_row_skipped`: verifies zero-unit rows do not cause divide-by-zero errors.
- `test_scenario_projection_matches_manual_calc`: verifies projected CGPA for hypothetical courses.
- `test_target_cgpa_achievable`: verifies required average grade point for a reachable target.
- `test_target_cgpa_unreachable`: verifies impossible targets return a clear not-achievable message.
- `test_empty_dataframe`: verifies empty data returns clean empty or zero results without crashing.

## Grade Scale Note

This app uses the Nigerian university 5-point grading scale:

```text
A = 5
B = 4
C = 3
D = 2
E = 1
F = 0
```

It does not use a 4.0 GPA scale.

## Known Limitations

The regression predictor is only a simple trend projection. It is not a guarantee of future performance.

The CSV is expected to use the column names `semester`, `course`, `units`, and `grade`. Rows with bad values are handled safely, but missing or renamed columns will be reported as invalid input.
