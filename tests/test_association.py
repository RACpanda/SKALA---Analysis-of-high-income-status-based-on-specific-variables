"""src.association의 v1.4 핵심 연관성 분석 흐름을 검증한다."""

from __future__ import annotations

import numpy as np

from src.association import AnalysisRequest, analyze_association


def test_continuous_association_without_controls(association_frame):
    result = analyze_association(
        association_frame,
        AnalysisRequest(exposure="age"),
    )
    analysis = result["analysis"]

    assert analysis["exposure_type"] == "continuous"
    assert analysis["unadjusted"]["method"] == "point_biserial_correlation"
    assert analysis["adjusted"]["method"] == "binary_logistic_regression"
    assert analysis["adjusted"]["adjustment_applied"] is False
    assert analysis["sample_size"] == len(association_frame)


def test_binary_association_with_one_control(association_frame):
    result = analyze_association(
        association_frame,
        AnalysisRequest(
            exposure="sex",
            controls=("age",),
        ),
    )
    analysis = result["analysis"]

    assert analysis["exposure_type"] == "binary"
    assert analysis["unadjusted"]["method"] == "binary_group_association"
    assert analysis["adjusted"]["adjustment_applied"] is True
    assert analysis["adjusted"]["controls"] == ["age"]
    assert analysis["psm"] is None


def test_categorical_association_with_multiple_controls(association_frame):
    result = analyze_association(
        association_frame,
        AnalysisRequest(
            exposure="education",
            controls=("age", "sex"),
        ),
    )
    analysis = result["analysis"]

    assert analysis["exposure_type"] == "categorical"
    assert analysis["unadjusted"]["method"] == "categorical_group_comparison"
    assert analysis["adjusted"]["adjustment_applied"] is True
    assert analysis["adjusted"]["controls"] == ["age", "sex"]
    assert len(analysis["unadjusted"]["groups"]) >= 2


def test_binary_association_with_optional_psm(association_frame):
    result = analyze_association(
        association_frame,
        AnalysisRequest(
            exposure="sex",
            controls=("age", "race"),
            include_psm=True,
        ),
    )
    psm = result["analysis"]["psm"]

    assert psm is not None
    assert psm["result"]["matching"]["matched_pairs"] >= 2
    assert len(psm["balance"]) > 0


def test_selected_control_missing_values_change_complete_case_sample(
    association_frame,
):
    df = association_frame.copy()
    missing_rows = 7
    df.loc[: missing_rows - 1, "age"] = np.nan

    result = analyze_association(
        df,
        AnalysisRequest(
            exposure="sex",
            controls=("age",),
        ),
    )

    assert result["analysis"]["rows_excluded_due_to_missing"] == missing_rows
    assert result["analysis"]["sample_size"] == len(df) - missing_rows
