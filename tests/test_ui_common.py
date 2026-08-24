"""공통 Streamlit 표시 helper의 계약을 검증한다."""

from __future__ import annotations

from src.ui_common import (
    category_value_label,
    format_p_value,
    format_percent,
    result_value_label,
    variable_label,
)


def test_variable_label():
    assert (
        variable_label(
            "age"
        )
        == "나이 (age)"
    )


def test_category_value_label():
    assert (
        category_value_label(
            "sex",
            "Male",
        )
        == "남성 (Male)"
    )


def test_unknown_category_keeps_original_value():
    assert (
        category_value_label(
            "sex",
            "UNKNOWN",
        )
        == "UNKNOWN"
    )


def test_result_value_label_uses_same_mapping():
    assert (
        result_value_label(
            "education",
            "Bachelors",
        )
        == "학사 (Bachelors)"
    )


def test_format_p_value():
    assert (
        format_p_value(
            0.123456
        )
        == "0.1235"
    )

    assert (
        format_p_value(
            0.0001
        )
        == "1.00e-04"
    )

    assert (
        format_p_value(
            None
        )
        == "추정 불가"
    )


def test_format_percent():
    assert (
        format_percent(
            0.1234
        )
        == "12.34%"
    )

    assert (
        format_percent(
            None
        )
        == "-"
    )