"""공통 사용자 표시용 라벨 계약을 검증한다."""

from __future__ import annotations

from src.labels import (
    CATEGORY_VALUE_LABELS,
    VARIABLE_LABELS,
    VARIABLE_TYPE_LABELS,
)


def test_core_variable_labels_are_defined():
    expected = {
        "age": "나이",
        "workclass": "고용 형태",
        "education": "교육 수준",
        "occupation": "직업",
        "native-country": "출신 국가",
    }

    for key, label in expected.items():
        assert VARIABLE_LABELS[key] == label


def test_south_is_not_arbitrarily_translated():
    assert (
        CATEGORY_VALUE_LABELS[
            "native-country"
        ]["South"]
        == "South"
    )


def test_variable_type_labels_are_centralized():
    assert VARIABLE_TYPE_LABELS == {
        "binary": "이진형",
        "continuous": "연속형",
        "categorical": "범주형",
    }