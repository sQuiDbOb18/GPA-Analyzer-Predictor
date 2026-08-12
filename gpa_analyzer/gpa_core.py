from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


GRADE_POINTS = {
    "A": 5.0,
    "B": 4.0,
    "C": 3.0,
    "D": 2.0,
    "E": 1.0,
    "F": 0.0,
}

POINT_TO_LETTER = [
    (4.5, "A"),
    (3.5, "B"),
    (2.5, "C"),
    (1.5, "D"),
    (0.5, "E"),
    (0.0, "F"),
]

REQUIRED_COLUMNS = ["semester", "course", "units", "grade"]


def _empty_clean_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_COLUMNS + ["grade_point", "quality_points"])


def validate_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | list[str]]]:
    """Validate a raw results dataframe and return clean rows plus a summary."""
    summary: dict[str, int | list[str]] = {
        "input_rows": int(len(df)),
        "dropped_rows": 0,
        "invalid_grade_rows": 0,
        "invalid_units_rows": 0,
        "missing_value_rows": 0,
        "missing_columns": [],
    }

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    summary["missing_columns"] = missing_columns
    if missing_columns:
        summary["dropped_rows"] = int(len(df))
        return _empty_clean_dataframe(), summary

    if df.empty:
        return _empty_clean_dataframe(), summary

    clean = df[REQUIRED_COLUMNS].copy()
    clean["grade"] = clean["grade"].astype("string").str.strip().str.upper()
    clean["semester"] = clean["semester"].astype("string").str.strip()
    clean["course"] = clean["course"].astype("string").str.strip()
    clean["units_numeric"] = pd.to_numeric(clean["units"], errors="coerce")

    missing_mask = clean[["semester", "course", "grade"]].isna().any(axis=1) | clean[
        ["semester", "course", "grade"]
    ].eq("").any(axis=1)
    invalid_grade_mask = ~clean["grade"].isin(GRADE_POINTS)
    invalid_units_mask = (
        clean["units_numeric"].isna()
        | (clean["units_numeric"] <= 0)
        | (clean["units_numeric"] % 1 != 0)
    )
    drop_mask = missing_mask | invalid_grade_mask | invalid_units_mask

    summary["missing_value_rows"] = int(missing_mask.sum())
    summary["invalid_grade_rows"] = int((invalid_grade_mask & ~missing_mask).sum())
    summary["invalid_units_rows"] = int((invalid_units_mask & ~missing_mask).sum())
    summary["dropped_rows"] = int(drop_mask.sum())

    valid = clean.loc[~drop_mask, ["semester", "course", "grade", "units_numeric"]].copy()
    if valid.empty:
        return _empty_clean_dataframe(), summary

    valid = valid.rename(columns={"units_numeric": "units"})
    valid["units"] = valid["units"].astype(int)
    valid["grade_point"] = valid["grade"].map(GRADE_POINTS).astype(float)
    valid["quality_points"] = valid["grade_point"] * valid["units"]
    return valid[REQUIRED_COLUMNS + ["grade_point", "quality_points"]], summary


def load_data(csv_path: str | Path = "results.csv") -> tuple[pd.DataFrame, dict[str, int | str | list[str]]]:
    """Load and validate a CSV file, returning a clean dataframe and status summary."""
    path = Path(csv_path)
    if not path.exists():
        return _empty_clean_dataframe(), {
            "input_rows": 0,
            "dropped_rows": 0,
            "invalid_grade_rows": 0,
            "invalid_units_rows": 0,
            "missing_value_rows": 0,
            "missing_columns": [],
            "error": f"File not found: {path}",
        }

    try:
        raw = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return _empty_clean_dataframe(), {
            "input_rows": 0,
            "dropped_rows": 0,
            "invalid_grade_rows": 0,
            "invalid_units_rows": 0,
            "missing_value_rows": 0,
            "missing_columns": [],
            "error": "CSV is empty.",
        }

    clean, summary = validate_data(raw)
    return clean, summary


def semester_order(df: pd.DataFrame) -> list[str]:
    if df.empty or "semester" not in df.columns:
        return []
    return list(pd.unique(df["semester"]))


def compute_gpa(df: pd.DataFrame, order: Iterable[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["semester", "total_units", "quality_points", "gpa"])

    semester_sequence = list(order) if order is not None else semester_order(df)
    grouped = (
        df.groupby("semester", sort=False, observed=False)
        .agg(total_units=("units", "sum"), quality_points=("quality_points", "sum"))
        .reset_index()
    )
    grouped = grouped[grouped["total_units"] > 0].copy()
    if grouped.empty:
        return pd.DataFrame(columns=["semester", "total_units", "quality_points", "gpa"])

    grouped["gpa"] = (grouped["quality_points"] / grouped["total_units"]).round(2)
    if semester_sequence:
        grouped["semester"] = pd.Categorical(grouped["semester"], categories=semester_sequence, ordered=True)
        grouped = grouped.sort_values("semester").reset_index(drop=True)
        grouped["semester"] = grouped["semester"].astype(str)
    return grouped[["semester", "total_units", "quality_points", "gpa"]]


def compute_cgpa(df: pd.DataFrame, order: Iterable[str] | None = None) -> tuple[pd.DataFrame, float]:
    gpa = compute_gpa(df, order=order)
    if gpa.empty:
        return pd.DataFrame(columns=["semester", "total_units", "quality_points", "gpa", "cumulative_units", "cumulative_quality_points", "cgpa"]), 0.0

    result = gpa.copy()
    result["cumulative_units"] = result["total_units"].cumsum()
    result["cumulative_quality_points"] = result["quality_points"].cumsum()
    result["cgpa"] = (result["cumulative_quality_points"] / result["cumulative_units"]).round(2)
    return result, float(result["cgpa"].iloc[-1])


def scenario_projection(
    current_df: pd.DataFrame,
    hypothetical_courses: list[dict[str, object]] | pd.DataFrame,
    future_semester: str = "Projected Next Semester",
) -> dict[str, float | int | dict[str, int | list[str]]]:
    current_summary, current_cgpa = compute_cgpa(current_df)
    current_units = int(current_summary["total_units"].sum()) if not current_summary.empty else 0
    current_quality_points = float(current_summary["quality_points"].sum()) if not current_summary.empty else 0.0

    future_raw = pd.DataFrame(hypothetical_courses)
    if future_raw.empty:
        future_clean = _empty_clean_dataframe()
        validation = {
            "input_rows": 0,
            "dropped_rows": 0,
            "invalid_grade_rows": 0,
            "invalid_units_rows": 0,
            "missing_value_rows": 0,
            "missing_columns": [],
        }
    else:
        if "semester" not in future_raw.columns:
            future_raw["semester"] = future_semester
        future_clean, validation = validate_data(future_raw)

    future_units = int(future_clean["units"].sum()) if not future_clean.empty else 0
    future_quality_points = float(future_clean["quality_points"].sum()) if not future_clean.empty else 0.0
    projected_units = current_units + future_units
    projected_quality_points = current_quality_points + future_quality_points
    projected_cgpa = round(projected_quality_points / projected_units, 2) if projected_units else 0.0

    return {
        "current_cgpa": round(current_cgpa, 2),
        "projected_cgpa": projected_cgpa,
        "delta": round(projected_cgpa - current_cgpa, 2),
        "future_units": future_units,
        "future_quality_points": future_quality_points,
        "validation": validation,
    }


def target_cgpa_advice(current_df: pd.DataFrame, target_cgpa: float, remaining_units: int) -> dict[str, object]:
    current_summary, current_cgpa = compute_cgpa(current_df)
    completed_units = int(current_summary["total_units"].sum()) if not current_summary.empty else 0
    completed_quality_points = float(current_summary["quality_points"].sum()) if not current_summary.empty else 0.0

    if remaining_units <= 0:
        achievable = round(current_cgpa, 2) >= round(target_cgpa, 2)
        message = (
            f"With no remaining units, your CGPA stays at {current_cgpa:.2f}."
            if achievable
            else f"A CGPA of {target_cgpa:.2f} is not achievable with no remaining units."
        )
        return {
            "achievable": achievable,
            "required_average": 0.0,
            "letter_grade": None,
            "message": message,
        }

    total_units_at_finish = completed_units + remaining_units
    required_total_quality_points = target_cgpa * total_units_at_finish
    required_remaining_quality_points = required_total_quality_points - completed_quality_points
    required_average = required_remaining_quality_points / remaining_units

    if required_average > 5.0:
        return {
            "achievable": False,
            "required_average": round(required_average, 2),
            "letter_grade": None,
            "message": (
                f"A CGPA of {target_cgpa:.2f} is not achievable over your remaining "
                f"{remaining_units} units because it would require averaging {required_average:.2f}, above 5.0."
            ),
        }

    required_average = max(required_average, 0.0)
    letter_grade = grade_point_to_letter(required_average)
    return {
        "achievable": True,
        "required_average": round(required_average, 2),
        "letter_grade": letter_grade,
        "message": (
            f"You need to average at least a {letter_grade} ({required_average:.2f}) over your "
            f"remaining {remaining_units} units to reach a CGPA of {target_cgpa:.2f}."
        ),
    }


def grade_point_to_letter(point: float) -> str:
    for threshold, letter in POINT_TO_LETTER:
        if point >= threshold:
            return letter
    return "F"


def trend_predict(cgpa_df: pd.DataFrame) -> dict[str, object]:
    if cgpa_df.empty or len(cgpa_df) < 2:
        return {
            "available": False,
            "message": "Need at least two semesters to build a trend projection.",
        }

    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        return {
            "available": False,
            "message": "Scikit-learn is not installed, so the optional trend projection was skipped.",
        }

    x = pd.DataFrame({"semester_index": range(1, len(cgpa_df) + 1)})
    y = cgpa_df["cgpa"].astype(float)
    model = LinearRegression()
    model.fit(x, y)
    next_index = len(cgpa_df) + 1
    predicted = float(model.predict(pd.DataFrame({"semester_index": [next_index]}))[0])
    predicted = max(0.0, min(5.0, predicted))
    return {
        "available": True,
        "next_semester_index": next_index,
        "projected_cgpa": round(predicted, 2),
        "message": (
            f"Trend projection only, not a guarantee: next semester CGPA may be about {predicted:.2f} "
            "if the current linear trend continues."
        ),
    }
