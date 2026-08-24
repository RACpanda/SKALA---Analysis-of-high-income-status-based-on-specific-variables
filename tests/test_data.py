"""src.data의 v1.4 공통 정제·분석별 결측 처리 계약을 검증한다."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import clean_data, prepare_analysis_data


def _sample_frame(**overrides) -> pd.DataFrame:
    base = {
        "age": [39, 45],
        "workclass": ["Private", "Private"],
        "fnlwgt": [77516, 83311],
        "education": ["Bachelors", "HS-grad"],
        "education-num": [13, 9],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Adm-clerical", "Exec-managerial"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "White"],
        "sex": ["Male", "Male"],
        "capital-gain": [2174, 0],
        "capital-loss": [0, 0],
        "hours-per-week": [40, 13],
        "native-country": ["United-States", "United-States"],
        "income": ["<=50K", ">50K"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_income_is_mapped_to_binary_high_income():
    cleaned, _ = clean_data(_sample_frame())

    assert cleaned["high_income"].astype(int).tolist() == [0, 1]


def test_trailing_period_and_whitespace_are_normalized_before_mapping():
    cleaned, _ = clean_data(
        _sample_frame(income=[" <=50K.", " >50K."])
    )

    assert cleaned["income"].tolist() == ["<=50K", ">50K"]
    assert cleaned["high_income"].astype(int).tolist() == [0, 1]


def test_missing_values_are_retained_in_common_cleaning_then_removed_per_analysis():
    cleaned, cleaning_info = clean_data(
        _sample_frame(workclass=["?", "Private"])
    )

    # v1.4 공통 정제는 결측이라는 이유만으로 행 전체를 제거하지 않는다.
    assert len(cleaned) == 2
    assert cleaned["workclass"].isna().sum() == 1
    assert cleaning_info["rows_with_missing"] == 1

    analysis = prepare_analysis_data(
        cleaned,
        ["workclass", "high_income"],
    )

    assert len(analysis) == 1
    assert analysis["workclass"].tolist() == ["Private"]


def test_duplicate_rows_are_removed():
    df = pd.concat(
        [_sample_frame(), _sample_frame()],
        ignore_index=True,
    )

    cleaned, cleaning_info = clean_data(df)

    assert len(cleaned) == 2
    assert cleaning_info["duplicate_removed"] == 2


def test_invalid_income_value_is_removed_as_logically_invalid():
    cleaned, cleaning_info = clean_data(
        _sample_frame(income=["<=50K", "unexpected-value"])
    )

    assert cleaned["income"].tolist() == ["<=50K"]
    assert cleaning_info["invalid_removed"] == 1


def test_out_of_range_numeric_value_is_removed():
    cleaned, cleaning_info = clean_data(
        _sample_frame(age=[39, 150])
    )

    assert cleaned["age"].tolist() == [39]
    assert cleaning_info["invalid_removed"] == 1


def test_numeric_conversion_failure_becomes_missing_but_row_is_retained():
    cleaned, cleaning_info = clean_data(
        _sample_frame(age=["not-a-number", 45])
    )

    assert len(cleaned) == 2
    assert cleaned["age"].isna().sum() == 1
    assert cleaning_info["rows_with_missing"] == 1


def test_prepare_analysis_data_rejects_missing_required_column():
    cleaned, _ = clean_data(_sample_frame())

    with pytest.raises(ValueError, match="분석에 필요한 열"):
        prepare_analysis_data(
            cleaned,
            ["age", "does-not-exist", "high_income"],
        )
