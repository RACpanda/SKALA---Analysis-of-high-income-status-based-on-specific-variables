"""Adult Census Income 데이터의 범용 탐색적 데이터 분석(EDA).

이 모듈은 서비스의 연관성 분석이나 예측을 직접 수행하지 않는다.
데이터 크기, 결측값, 중복, 변수 분포와 target 분포를 확인하기 위한
개발·데이터 품질 점검용 보조 모듈이다.

df가 전달되지 않으면 load_before_cleaning()을 사용해
행 제거 전 데이터를 점검한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
)

from src.data import load_before_cleaning

# ============================================================
# 입력 검증
# ============================================================

def validate_eda_input(df: pd.DataFrame,) -> None:
    """EDA 입력 DataFrame의 기본 구조를 검증한다."""

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "EDA 입력은 pandas.DataFrame이어야 합니다. "
            f"현재 타입: {type(df).__name__}"
        )

    if df.empty:
        raise ValueError(
            "EDA 입력 데이터가 비어 있습니다."
        )

    duplicate_columns = (
        pd.Index(df.columns)
        .duplicated(keep=False)
    )

    if duplicate_columns.any():
        duplicated = sorted(
            set(
                pd.Index(
                    df.columns
                )[
                    duplicate_columns
                ]
                .astype(str)
                .tolist()
            )
        )

        raise ValueError(
            "EDA 입력에 중복 열 이름이 있습니다: "
            f"{duplicated}"
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"EDA 입력에 target '{TARGET_COLUMN}'이 없습니다."
        )

    invalid_target_mask = (
        df[TARGET_COLUMN].notna()
        & ~df[TARGET_COLUMN].isin(
            [0, 1]
        )
    )

    if invalid_target_mask.any():
        invalid_values = (
            df.loc[
                invalid_target_mask,
                TARGET_COLUMN,
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{TARGET_COLUMN}은 0과 1 또는 결측값만 "
            "포함해야 합니다. "
            f"현재 값: {invalid_values}"
        )

# ============================================================
# 데이터 품질
# ============================================================

def build_missing_values_table(df: pd.DataFrame,) -> pd.DataFrame:
    """열별 결측값 개수와 비율을 계산한다."""

    total_rows = len(df)

    result = pd.DataFrame(
        {
            "column": (
                df.columns
                .astype(str)
            ),
            "dtype": [
                str(dtype)
                for dtype in df.dtypes
            ],
            "missing_count": (
                df.isna()
                .sum()
                .astype(int)
                .to_numpy()
            ),
        }
    )

    result["missing_rate"] = (
        result["missing_count"]
        / total_rows
    )

    return (
        result
        .sort_values(
            [
                "missing_count",
                "column",
            ],
            ascending=[
                False,
                True,
            ],
            ignore_index=True,
        )
    )


def build_duplicate_result(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """완전 중복 행의 개수와 비율을 계산한다."""

    row_count = len(df)

    duplicate_rows = int(
        df.duplicated().sum()
    )

    return pd.DataFrame(
        [
            {
                "row_count": (
                    row_count
                ),
                "duplicate_rows": (
                    duplicate_rows
                ),
                "duplicate_rate": (
                    duplicate_rows
                    / row_count
                ),
            }
        ]
    )

# ============================================================
# 변수별 기술통계
# ============================================================

def build_descriptive_stats(df: pd.DataFrame,) -> pd.DataFrame:
    """수치형 변수의 기본 기술통계를 생성한다."""

    numeric_df = (df.select_dtypes(include="number"))

    if numeric_df.empty:
        return pd.DataFrame(
            columns=[
                "column",
                "count",
                "mean",
                "std",
                "min",
                "q1",
                "median",
                "q3",
                "max",
                "missing_count",
            ]
        )

    stats = (
        numeric_df
        .describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
            ]
        )
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index": "column",
                "25%": "q1",
                "50%": "median",
                "75%": "q3",
            }
        )
    )

    stats["missing_count"] = (
        stats["column"]
        .map(
            numeric_df
            .isna()
            .sum()
        )
        .astype(int)
    )

    return stats[
        [
            "column",
            "count",
            "mean",
            "std",
            "min",
            "q1",
            "median",
            "q3",
            "max",
            "missing_count",
        ]
    ]


def build_categorical_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """범주형 변수별 범주 빈도와 비율을 계산한다."""

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "string",
                "category",
                "bool",
            ]
        )
        .columns
        .tolist()
    )

    records: list[dict] = []

    for column in categorical_columns:
        counts = (
            df[column]
            .astype("string")
            .value_counts(
                dropna=False
            )
        )

        for category, count in (
            counts.items()
        ):
            records.append(
                {
                    "column": column,
                    "category": (
                        "<NA>"
                        if pd.isna(category)
                        else str(category)
                    ),
                    "count": int(
                        count
                    ),
                    "proportion": float(
                        count / len(df)
                    ),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "column",
                "category",
                "count",
                "proportion",
            ]
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "column",
                "count",
                "category",
            ],
            ascending=[
                True,
                False,
                True,
            ],
            ignore_index=True,
        )
    )

# ============================================================
# Target 분포
# ============================================================

def build_target_summary(df: pd.DataFrame,) -> dict[str, int | float]:
    """고소득 target의 유효 표본 수, 결측 수와 클래스 분포를 계산한다."""

    target = df[TARGET_COLUMN]

    valid_count = int(target.count())
    missing_count = int(target.isna().sum())
    high_income_count = int(target.eq(1).sum())
    low_income_count = int(target.eq(0).sum())

    high_income_rate = (
        float(
            high_income_count
            / valid_count
        )
        if valid_count
        else 0.0
    )

    return {
        "valid_count": valid_count,
        "missing_count": missing_count,
        "low_income_count": (
            low_income_count
        ),
        "high_income_count": (
            high_income_count
        ),
        "high_income_rate": (
            high_income_rate
        ),
    }


# ============================================================
# 전체 EDA
# ============================================================

def run_eda(
    df: pd.DataFrame | None = None,
    *,
    data_path: str | Path = RAW_DATA_PATH,
) -> dict:
    """Adult 데이터의 범용 EDA 결과를 메모리 객체로 반환한다.

    df가 전달되지 않으면 문자열·숫자 표현만 정규화한
    행 제거 전 데이터를 load_before_cleaning()으로 불러온다.

    Returns:
        summary:
            데이터 크기, 데이터 품질과 target 분포 요약.

        tables:
            결측값, 중복, 수치형 기술통계와 범주형 분포 DataFrame.
    """

    if df is None:
        analysis_df = (load_before_cleaning(path=data_path,))
    else:
        analysis_df = df.copy()

    validate_eda_input(analysis_df)

    missing_values = (build_missing_values_table(analysis_df))
    duplicate_result = (build_duplicate_result(analysis_df))
    descriptive_stats = (build_descriptive_stats(analysis_df))
    categorical_summary = ( build_categorical_summary(analysis_df))
    target_summary = (build_target_summary(analysis_df))

    summary = {
        "rows": int(
            len(analysis_df)
        ),
        "columns": int(
            len(analysis_df.columns)
        ),
        "missing_cells": int(
            analysis_df
            .isna()
            .sum()
            .sum()
        ),
        "columns_with_missing": int(
            analysis_df
            .isna()
            .any()
            .sum()
        ),
        "duplicate_rows": int(
            duplicate_result.loc[
                0,
                "duplicate_rows",
            ]
        ),
        "target": target_summary,
    }

    return {
        "summary": summary,
        "tables": {
            "missing_values": (
                missing_values
            ),
            "duplicate_result": (
                duplicate_result
            ),
            "descriptive_stats": (
                descriptive_stats
            ),
            "categorical_summary": (
                categorical_summary
            ),
        },
    }