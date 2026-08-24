"""Streamlit 사용자 화면에서 공통으로 사용하는 표시 helper.

분석이나 모델 계산은 수행하지 않고,
라벨 변환과 숫자 표시 등 presentation 역할만 담당한다.
"""

from __future__ import annotations

import streamlit as st

from src.labels import (
    CATEGORY_VALUE_LABELS,
    VARIABLE_LABELS,
)


def category_value_label(
    feature: str,
    value,
) -> str:
    """범주 값을 '한글 (원본값)' 형식으로 표시한다."""

    korean = (
        CATEGORY_VALUE_LABELS
        .get(feature, {})
        .get(value)
    )

    if korean is None:
        return str(value)

    return f"{korean} ({value})"


def variable_label(
    variable: str,
) -> str:
    """변수명을 '한글 (원본 컬럼명)' 형식으로 표시한다."""

    korean = VARIABLE_LABELS.get(
        variable,
        variable,
    )

    return f"{korean} ({variable})"


def format_p_value(
    value: float | None,
) -> str:
    """p-value를 사용자 화면용 문자열로 변환한다."""

    if value is None:
        return "추정 불가"

    numeric_value = float(value)

    if numeric_value < 0.001:
        return f"{numeric_value:.2e}"

    return f"{numeric_value:.4f}"


def format_percent(
    value: float | None,
) -> str:
    """0~1 비율을 백분율 문자열로 변환한다."""

    if value is None:
        return "-"

    return (
        f"{float(value) * 100:.2f}%"
    )


def result_value_label(
    variable: str,
    value,
) -> str:
    """분석 결과의 범주값을 사용자 표시용 이름으로 변환한다."""

    korean = (
        CATEGORY_VALUE_LABELS
        .get(variable, {})
        .get(value)
    )

    if korean is None:
        return str(value)

    return f"{korean} ({value})"


def display_interpretation_note(
    text: str,
) -> None:
    """해석상 주의 문구를 공통 스타일로 표시한다."""

    st.markdown(
        (
            '<div class="interpretation-box">'
            f"{text}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )