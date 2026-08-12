import pandas as pd

from gpa_core import (
    compute_cgpa,
    compute_gpa,
    scenario_projection,
    target_cgpa_advice,
    validate_data,
)


def clean(rows):
    raw = pd.DataFrame(rows)
    validated, _ = validate_data(raw)
    return validated


def test_gpa_single_semester():
    df = clean(
        [
            {"semester": "100L First", "course": "MTH101", "units": 3, "grade": "A"},
            {"semester": "100L First", "course": "GST101", "units": 2, "grade": "B"},
        ]
    )

    gpa = compute_gpa(df)

    assert gpa.loc[0, "gpa"] == 4.6


def test_cgpa_across_semesters():
    df = clean(
        [
            {"semester": "100L First", "course": "MTH101", "units": 3, "grade": "A"},
            {"semester": "100L First", "course": "GST101", "units": 2, "grade": "B"},
            {"semester": "100L Second", "course": "MTH102", "units": 3, "grade": "C"},
            {"semester": "100L Second", "course": "GST102", "units": 2, "grade": "B"},
        ]
    )

    cgpa_table, final_cgpa = compute_cgpa(df)

    assert cgpa_table.loc[0, "cgpa"] == 4.6
    assert final_cgpa == 4.0


def test_invalid_grade_dropped():
    raw = pd.DataFrame(
        [
            {"semester": "100L First", "course": "MTH101", "units": 3, "grade": "A"},
            {"semester": "100L First", "course": "GST101", "units": 2, "grade": "Z"},
        ]
    )

    clean_df, summary = validate_data(raw)

    assert len(clean_df) == 1
    assert summary["invalid_grade_rows"] == 1
    assert summary["dropped_rows"] == 1


def test_zero_units_row_skipped():
    raw = pd.DataFrame(
        [
            {"semester": "100L First", "course": "MTH101", "units": 0, "grade": "A"},
            {"semester": "100L First", "course": "GST101", "units": 2, "grade": "B"},
        ]
    )

    clean_df, summary = validate_data(raw)
    gpa = compute_gpa(clean_df)

    assert summary["invalid_units_rows"] == 1
    assert gpa.loc[0, "total_units"] == 2
    assert gpa.loc[0, "gpa"] == 4.0


def test_scenario_projection_matches_manual_calc():
    df = clean(
        [
            {"semester": "100L First", "course": "MTH101", "units": 3, "grade": "A"},
            {"semester": "100L First", "course": "GST101", "units": 2, "grade": "B"},
        ]
    )

    projection = scenario_projection(
        df,
        [
            {"course": "CSC102", "units": 3, "grade": "C"},
            {"course": "GST102", "units": 2, "grade": "A"},
        ],
    )

    assert projection["projected_cgpa"] == 4.2


def test_target_cgpa_achievable():
    df = clean(
        [
            {"semester": "100L First", "course": "MTH101", "units": 10, "grade": "B"},
        ]
    )

    advice = target_cgpa_advice(df, target_cgpa=4.2, remaining_units=10)

    assert advice["achievable"] is True
    assert advice["required_average"] == 4.4
    assert "You need to average" in advice["message"]


def test_target_cgpa_unreachable():
    df = clean(
        [
            {"semester": "100L First", "course": "MTH101", "units": 100, "grade": "C"},
        ]
    )

    advice = target_cgpa_advice(df, target_cgpa=4.9, remaining_units=10)

    assert advice["achievable"] is False
    assert "not achievable" in advice["message"]


def test_empty_dataframe():
    clean_df, summary = validate_data(pd.DataFrame(columns=["semester", "course", "units", "grade"]))
    gpa = compute_gpa(clean_df)
    cgpa_table, final_cgpa = compute_cgpa(clean_df)

    assert summary["input_rows"] == 0
    assert gpa.empty
    assert cgpa_table.empty
    assert final_cgpa == 0.0
