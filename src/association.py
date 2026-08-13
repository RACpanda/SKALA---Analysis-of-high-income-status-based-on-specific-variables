"""Adult Census Income 범용 연관성 분석.

사용자가 선택한 관심 변수 1개와 통제 변수 0개 이상을 이용해
고소득 여부와의 조정 전·후 연관성을 분석한다.

주요 역할:
    1. 분석 요청 검증
    2. 분석에 필요한 표본 구성
    3. 관심 변수 유형에 맞는 조정 전 분석
    4. 통제변수를 포함한 Logistic Regression
    5. 선택적으로 이진 관심 변수에 대한 PSM 수행
    6. 웹에서 사용할 하나의 결과 객체 반환

결과변수는 서비스 목적에 따라 high_income으로 고정한다.

주의:
    Logistic Regression이나 PSM에서 통제변수를 사용하더라도
    관측되지 않은 교란요인은 통제할 수 없다.
    따라서 결과는 조건부 연관성으로 해석하며
    확정적인 인과효과를 의미하지 않는다.

    modeling.py와 목적이 다르다.
    - association.py: 변수와 고소득 여부의 관계 설명
    - modeling.py: 새로운 입력 조건의 고소득 여부 예측
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import (
    chi2_contingency,
    pointbiserialr,
)
from statsmodels.tools.sm_exceptions import (
    PerfectSeparationError,
)

from src.config import (
    ANALYSIS_VARIABLE_TYPES,
    ANALYSIS_VARIABLES,
    TARGET_COLUMN,
)
from src.data import prepare_analysis_data
from src.statistics import (
    StatisticsError,
    binary_group_association,
    propensity_score_matching,
)


VariableType = Literal[
    "binary",
    "continuous",
    "categorical",
]


class AssociationError(ValueError):
    """연관성 분석 요청 또는 모델 적합 과정에서 발생하는 오류."""


@dataclass(frozen=True)
class AnalysisRequest:
    """사용자가 요청한 고소득 연관성 분석 조건.

    Args:
        exposure:
            고소득 여부와의 관계를 직접 확인할 관심 변수.

        controls:
            관심 변수와 고소득 여부의 관계를 분석할 때
            통계적으로 함께 조정할 변수.

        include_psm:
            True이면 조건이 충족되는 경우 PSM을 추가 수행한다.
            현재 PSM은 이진 관심 변수와 하나 이상의 통제변수가
            있는 경우에만 지원한다.
    """

    exposure: str
    controls: tuple[str, ...] = ()
    include_psm: bool = False


# ============================================================
# 요청 검증
# ============================================================

def _validate_request(
    df: pd.DataFrame,
    request: AnalysisRequest,
) -> None:
    """서비스에서 허용하는 연관성 분석 요청인지 확인한다."""

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise AssociationError(
            "연관성 분석 입력은 "
            "pandas.DataFrame이어야 합니다."
        )

    if df.empty:
        raise AssociationError(
            "연관성 분석 데이터가 비어 있습니다."
        )

    if (
        request.exposure
        not in ANALYSIS_VARIABLES
    ):
        raise AssociationError(
            "관심 변수로 사용할 수 없는 변수입니다: "
            f"{request.exposure}"
        )

    invalid_controls = [
        control
        for control in request.controls
        if control
        not in ANALYSIS_VARIABLES
    ]

    if invalid_controls:
        raise AssociationError(
            "통제 변수로 사용할 수 없는 변수가 있습니다: "
            f"{invalid_controls}"
        )

    if (
        request.exposure
        in request.controls
    ):
        raise AssociationError(
            f"관심 변수 '{request.exposure}'를 "
            "통제 변수에 동시에 포함할 수 없습니다."
        )

    if (
        len(
            set(
                request.controls
            )
        )
        != len(
            request.controls
        )
    ):
        raise AssociationError(
            "통제 변수에 중복된 변수가 있습니다."
        )

    required_columns = [
        TARGET_COLUMN,
        request.exposure,
        *request.controls,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise AssociationError(
            "연관성 분석에 필요한 변수가 없습니다: "
            f"{missing_columns}"
        )

    if request.include_psm:
        exposure_type = (
            ANALYSIS_VARIABLE_TYPES[
                request.exposure
            ]
        )

        if exposure_type != "binary":
            raise AssociationError(
                "현재 PSM은 이진 관심 변수에만 사용할 수 있습니다. "
                f"'{request.exposure}'의 유형은 "
                f"'{exposure_type}'입니다."
            )

        if not request.controls:
            raise AssociationError(
                "PSM을 사용하려면 최소 1개의 "
                "통제 변수가 필요합니다."
            )


# ============================================================
# Target 검증
# ============================================================

def _validate_binary_target(
    series: pd.Series,
) -> None:
    """high_income이 정확히 0과 1을 모두 포함하는지 확인한다."""

    try:
        numeric = pd.to_numeric(
            series.dropna(),
            errors="raise",
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise AssociationError(
            f"결과 변수 '{TARGET_COLUMN}'을 "
            "0/1 숫자로 해석할 수 없습니다."
        ) from exc

    values = set(
        numeric.unique()
    )

    if values != {
        0,
        1,
    }:
        raise AssociationError(
            f"결과 변수 '{TARGET_COLUMN}'은 "
            "0과 1을 모두 포함해야 합니다. "
            f"현재 값: {values}"
        )


# ============================================================
# 범주 처리
# ============================================================

def _ordered_levels(
    series: pd.Series,
) -> list:
    """관측된 범주를 재현 가능한 순서로 정렬한다."""

    return sorted(
        series
        .dropna()
        .unique()
        .tolist(),
        key=lambda value: str(
            value
        ),
    )


def _binary_levels(
    series: pd.Series,
) -> tuple:
    """이진 변수의 기준 범주와 비교 범주를 반환한다."""

    levels = _ordered_levels(
        series
    )

    if len(levels) != 2:
        raise AssociationError(
            f"이진 변수 '{series.name}'에는 "
            "정확히 두 범주가 필요합니다. "
            f"현재 범주 수: {len(levels)}"
        )

    return (
        levels[0],
        levels[1],
    )


# ============================================================
# 조정 전 분석
# ============================================================

def _unadjusted_summary(
    df: pd.DataFrame,
    request: AnalysisRequest,
    exposure_type: VariableType,
) -> dict:
    """통제변수를 적용하기 전 관심 변수와 고소득 여부의 관계를 계산한다."""

    exposure = (
        request.exposure
    )

    # 이진 관심 변수의 집단 비교는 statistics.py의
    # 공통 통계 함수를 사용해 계산 중복을 방지한다.
    if exposure_type == "binary":
        try:
            result = (
                binary_group_association(
                    df,
                    exposure=exposure,
                    outcome=TARGET_COLUMN,
                )
            )
        except StatisticsError as exc:
            raise AssociationError(
                str(exc)
            ) from exc

        return result[
            "analysis"
        ]

    # --------------------------------------------------------
    # 연속형 관심 변수
    # --------------------------------------------------------
    if exposure_type == "continuous":
        exposure_values = (
            pd.to_numeric(
                df[exposure],
                errors="raise",
            )
            .astype(float)
        )

        if exposure_values.nunique() < 2:
            raise AssociationError(
                f"관심 변수 '{exposure}'에 "
                "서로 다른 값이 2개 이상 필요합니다."
            )

        target_values = (
            df[TARGET_COLUMN]
            .astype(int)
        )

        correlation, p_value = (
            pointbiserialr(
                target_values,
                exposure_values,
            )
        )

        # 그래프도 동일한 분석 표본을 사용하도록
        # 연속형 관심 변수의 구간별 실제 고소득률을 함께 계산한다.
        bin_count = min(
            10,
            int(
                exposure_values.nunique()
            ),
        )

        binned = pd.DataFrame(
            {
                exposure: exposure_values,
                TARGET_COLUMN: target_values,
            }
        )

        binned["__bin__"] = pd.qcut(
            binned[exposure],
            q=bin_count,
            duplicates="drop",
        )

        bins = (
            binned
            .groupby(
                "__bin__",
                observed=True,
            )
            .agg(
                exposure_mean=(
                    exposure,
                    "mean",
                ),
                exposure_min=(
                    exposure,
                    "min",
                ),
                exposure_max=(
                    exposure,
                    "max",
                ),
                n=(
                    TARGET_COLUMN,
                    "size",
                ),
                target_rate=(
                    TARGET_COLUMN,
                    "mean",
                ),
            )
            .reset_index(drop=True)
            .sort_values(
                "exposure_mean"
            )
        )

        return {
            "method": (
                "point_biserial_correlation"
            ),
            "correlation": float(
                correlation
            ),
            "p_value": float(
                p_value
            ),
            "exposure_mean": float(
                exposure_values.mean()
            ),
            "exposure_std": float(
                exposure_values.std(
                    ddof=1
                )
            ),
            "bins": (
                bins.to_dict(
                    orient="records"
                )
            ),
        }

    # --------------------------------------------------------
    # 다범주형 관심 변수
    # --------------------------------------------------------

    grouped = (
        df
        .groupby(
            exposure,
            observed=True,
        )[TARGET_COLUMN]
        .agg(
            n="size",
            target_rate="mean",
        )
        .reset_index()
        .sort_values(
            "target_rate",
            ascending=False,
            ignore_index=True,
        )
    )

    if len(grouped) < 2:
        raise AssociationError(
            f"관심 변수 '{exposure}'에 "
            "서로 다른 범주가 2개 이상 필요합니다."
        )

    contingency = pd.crosstab(
        df[exposure],
        df[TARGET_COLUMN],
    )

    (
        chi2,
        p_value,
        degrees_of_freedom,
        expected,
    ) = chi2_contingency(
        contingency
    )

    expected = np.asarray(
        expected,
        dtype=float,
    )

    return {
        "method": (
            "categorical_group_comparison"
        ),
        "chi2_statistic": float(
            chi2
        ),
        "chi2_p_value": float(
            p_value
        ),
        "degrees_of_freedom": int(
            degrees_of_freedom
        ),
        "minimum_expected_count": float(
            expected.min()
        ),
        "expected_cells_under_5": int(
            (
                expected < 5
            ).sum()
        ),
        "groups": (
            grouped.to_dict(
                orient="records"
            )
        ),
    }


# ============================================================
# Logistic Regression 입력 행렬
# ============================================================

def _build_design_matrix(
    df: pd.DataFrame,
    request: AnalysisRequest,
    exposure_type: VariableType,
) -> tuple[
    pd.DataFrame,
    list[str],
    dict,
]:
    """관심 변수와 통제 변수를 Logistic Regression 입력으로 변환한다.

    변수 유형은 DataFrame dtype을 추측하지 않고
    config.py의 ANALYSIS_VARIABLE_TYPES를 기준으로 처리한다.

    처리 원칙:
        continuous:
            숫자값 그대로 사용한다.

        binary:
            기준 범주를 0, 비교 범주를 1로 변환한다.

        categorical:
            첫 범주를 기준으로 두고 one-hot encoding한다.
    """

    exposure = (
        request.exposure
    )

    controls = list(
        request.controls
    )

    predictors = [
        exposure,
        *controls,
    ]

    X = (
        df[
            predictors
        ]
        .copy()
    )

    categorical_columns: list[str] = []

    exposure_metadata: dict = {
        "type": exposure_type,
    }

    # --------------------------------------------------------
    # 관심 변수
    # --------------------------------------------------------

    if exposure_type == "binary":
        (
            reference,
            comparison,
        ) = _binary_levels(
            X[exposure]
        )

        X[exposure] = (
            X[exposure]
            .map(
                {
                    reference: 0.0,
                    comparison: 1.0,
                }
            )
            .astype(float)
        )

        exposure_metadata.update(
            {
                "reference_level": (
                    reference
                ),
                "comparison_level": (
                    comparison
                ),
            }
        )

    elif exposure_type == "continuous":
        X[exposure] = (
            pd.to_numeric(
                X[exposure],
                errors="raise",
            )
            .astype(float)
        )

        if (
            X[exposure].nunique()
            < 2
        ):
            raise AssociationError(
                f"관심 변수 '{exposure}'에 "
                "서로 다른 값이 2개 이상 필요합니다."
            )

        exposure_metadata[
            "interpretation_unit"
        ] = 1

    else:
        levels = _ordered_levels(
            X[exposure]
        )

        if len(levels) < 2:
            raise AssociationError(
                f"관심 변수 '{exposure}'에 "
                "서로 다른 범주가 2개 이상 필요합니다."
            )

        X[exposure] = (
            pd.Categorical(
                X[exposure],
                categories=levels,
            )
        )

        categorical_columns.append(
            exposure
        )

        exposure_metadata.update(
            {
                "reference_level": (
                    levels[0]
                ),
                "levels": (
                    levels
                ),
            }
        )

    # --------------------------------------------------------
    # 통제 변수
    # --------------------------------------------------------

    for column in controls:
        variable_type = (
            ANALYSIS_VARIABLE_TYPES[
                column
            ]
        )

        if variable_type == "continuous":
            X[column] = (
                pd.to_numeric(
                    X[column],
                    errors="raise",
                )
                .astype(float)
            )

        elif variable_type == "binary":
            levels = _ordered_levels(
                X[column]
            )

            # 이번 분석 표본에서 한 범주만 남은 binary control은
            # 이후 constant column 제거 단계에서 삭제한다.
            if len(levels) == 1:
                X[column] = 0.0

            elif len(levels) == 2:
                X[column] = (
                    X[column]
                    .map(
                        {
                            levels[0]: 0.0,
                            levels[1]: 1.0,
                        }
                    )
                    .astype(float)
                )

            else:
                raise AssociationError(
                    f"binary로 설정된 통제 변수 '{column}'에 "
                    f"{len(levels)}개의 범주가 있습니다."
                )

        else:
            levels = _ordered_levels(
                X[column]
            )

            if not levels:
                raise AssociationError(
                    f"통제 변수 '{column}'에 "
                    "사용 가능한 값이 없습니다."
                )

            X[column] = (
                pd.Categorical(
                    X[column],
                    categories=levels,
                )
            )

            categorical_columns.append(
                column
            )

    if categorical_columns:
        X = pd.get_dummies(
            X,
            columns=categorical_columns,
            drop_first=True,
            dtype=float,
        )

    # --------------------------------------------------------
    # 관심 변수 회귀계수 이름
    # --------------------------------------------------------

    if exposure_type in {
        "binary",
        "continuous",
    }:
        exposure_terms = [
            exposure
        ]

    else:
        prefix = (
            f"{exposure}_"
        )

        exposure_terms = [
            column
            for column in X.columns
            if column.startswith(
                prefix
            )
        ]

    # 이번 분석 표본에서 값이 하나뿐인 통제변수는
    # 회귀모형에 정보를 제공하지 않으므로 제거한다.
    constant_columns = [
        column
        for column in X.columns
        if (
            X[column]
            .nunique(
                dropna=False
            )
            <= 1
        )
    ]

    if constant_columns:
        X = X.drop(
            columns=constant_columns
        )

        exposure_terms = [
            term
            for term in exposure_terms
            if term in X.columns
        ]

    if not exposure_terms:
        raise AssociationError(
            f"관심 변수 '{exposure}'의 "
            "회귀계수를 생성할 수 없습니다."
        )

    if X.empty:
        raise AssociationError(
            "Logistic Regression에 사용할 "
            "설명변수가 없습니다."
        )

    return (
        X.astype(float),
        exposure_terms,
        exposure_metadata,
    )


# ============================================================
# 조정 후 Logistic Regression
# ============================================================

def _fit_logistic_association(
    df: pd.DataFrame,
    request: AnalysisRequest,
    exposure_type: VariableType,
) -> dict:
    """관심 변수와 통제 변수를 포함한 이항 Logistic Regression을 수행한다."""

    (
        X,
        exposure_terms,
        exposure_metadata,
    ) = _build_design_matrix(
        df,
        request,
        exposure_type,
    )

    y = (
        df[
            TARGET_COLUMN
        ]
        .astype(int)
    )

    X = sm.add_constant(
        X,
        has_constant="add",
    )

    matrix_rank = (
        np.linalg.matrix_rank(
            X.to_numpy(
                dtype=float
            )
        )
    )

    if (
        matrix_rank
        < X.shape[1]
    ):
        raise AssociationError(
            "설명변수 사이에 완전한 선형 종속성이 있습니다. "
            "서로 중복되거나 동일한 정보를 가진 통제변수를 "
            "제거한 뒤 다시 분석하세요."
        )

    try:
        model = sm.Logit(
            y,
            X,
        )

        fitted = model.fit(
            disp=False,
            maxiter=200,
        )

    except PerfectSeparationError as exc:
        raise AssociationError(
            "일부 변수 조합이 고소득 여부를 완전히 분리하여 "
            "Logistic Regression을 추정할 수 없습니다."
        ) from exc

    except np.linalg.LinAlgError as exc:
        raise AssociationError(
            "Logistic Regression 행렬 계산에 실패했습니다. "
            "중복되거나 지나치게 유사한 통제변수를 확인하세요."
        ) from exc

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise AssociationError(
            "Logistic Regression 적합에 실패했습니다: "
            f"{exc}"
        ) from exc

    converged = bool(
        fitted.mle_retvals.get(
            "converged",
            True,
        )
    )

    if not converged:
        raise AssociationError(
            "Logistic Regression이 수렴하지 않았습니다. "
            "변수 조합이나 희소한 범주를 확인하세요."
        )

    confidence_interval = (
        fitted.conf_int(
            alpha=0.05
        )
    )

    exposure_results: list[dict] = []

    for term in exposure_terms:
        coefficient = float(
            fitted.params[
                term
            ]
        )

        ci_low_log = float(
            confidence_interval.loc[
                term,
                0,
            ]
        )

        ci_high_log = float(
            confidence_interval.loc[
                term,
                1,
            ]
        )

        if not np.isfinite(
            [
                coefficient,
                ci_low_log,
                ci_high_log,
            ]
        ).all():
            raise AssociationError(
                "관심 변수의 회귀계수 또는 신뢰구간이 "
                "유한한 값으로 추정되지 않았습니다. "
                "희소한 범주나 과도한 분리를 확인하세요."
            )

        odds_ratio = float(
            np.exp(
                coefficient
            )
        )

        ci_low = float(
            np.exp(
                ci_low_log
            )
        )

        ci_high = float(
            np.exp(
                ci_high_log
            )
        )

        if not np.isfinite(
            [
                odds_ratio,
                ci_low,
                ci_high,
            ]
        ).all():
            raise AssociationError(
                "Odds Ratio가 지나치게 커 안정적으로 "
                "표현할 수 없습니다. "
                "희소한 범주나 완전분리를 확인하세요."
            )

        exposure_results.append(
            {
                "term": term,
                "coefficient": (
                    coefficient
                ),
                "odds_ratio": (
                    odds_ratio
                ),
                "p_value": float(
                    fitted.pvalues[
                        term
                    ]
                ),
                "ci_95_low": (
                    ci_low
                ),
                "ci_95_high": (
                    ci_high
                ),
            }
        )

    return {
        "method": (
            "binary_logistic_regression"
        ),
        "adjustment_applied": bool(
            request.controls
        ),
        "controls": list(
            request.controls
        ),
        "exposure_metadata": (
            exposure_metadata
        ),
        "exposure_effects": (
            exposure_results
        ),
        "model_diagnostics": {
            "converged": (
                converged
            ),
            "n_observations": int(
                fitted.nobs
            ),
            "pseudo_r_squared": float(
                fitted.prsquared
            ),
            "aic": float(
                fitted.aic
            ),
            "bic": float(
                fitted.bic
            ),
        },
    }


# ============================================================
# PSM 결과 직렬화
# ============================================================

def _serializable_records(
    df: pd.DataFrame,
) -> list[dict]:
    """DataFrame을 웹/API 응답에 안전한 기본 Python 값으로 변환한다."""

    records: list[dict] = []

    for row in df.to_dict(
        orient="records"
    ):
        converted: dict = {}

        for (
            key,
            value,
        ) in row.items():
            if isinstance(
                value,
                np.generic,
            ):
                value = (
                    value.item()
                )

            if (
                isinstance(
                    value,
                    float,
                )
                and not np.isfinite(
                    value
                )
            ):
                value = None

            elif pd.isna(
                value
            ):
                value = None

            converted[
                key
            ] = value

        records.append(
            converted
        )

    return records


def _run_optional_psm(
    df: pd.DataFrame,
    request: AnalysisRequest,
) -> dict | None:
    """요청된 경우 이진 관심 변수의 PSM을 수행한다."""

    if not request.include_psm:
        return None

    try:
        (
            _,
            balance,
            result,
        ) = propensity_score_matching(
            df,
            exposure=request.exposure,
            covariates=list(
                request.controls
            ),
            outcome=TARGET_COLUMN,
        )

    except StatisticsError as exc:
        raise AssociationError(
            f"PSM 수행에 실패했습니다: {exc}"
        ) from exc

    return {
        "result": result,
        "balance": (
            _serializable_records(
                balance
            )
        ),
    }


# ============================================================
# 전체 연관성 분석
# ============================================================

def analyze_association(
    df: pd.DataFrame,
    request: AnalysisRequest,
) -> dict:
    """한 번의 사용자 요청에 대한 전체 고소득 연관성 분석을 수행한다.

    실행 흐름:
        1. 분석 요청 검증
        2. 이번 분석에 필요한 변수만 선택
        3. 해당 변수들의 결측 행 제거
        4. high_income 0/1 구조 검증
        5. 관심 변수 유형에 맞는 조정 전 분석
        6. Logistic Regression
        7. 요청한 경우 PSM
        8. 요청과 결과를 하나의 객체로 반환

    분석별 complete-case 방식을 사용하므로 사용자가 선택한
    통제 변수에 따라 최종 표본 수가 달라질 수 있다.
    """

    _validate_request(
        df,
        request,
    )

    required_columns = [
        TARGET_COLUMN,
        request.exposure,
        *request.controls,
    ]

    analysis_df = (
        prepare_analysis_data(
            df,
            required_columns,
        )
    )

    if analysis_df.empty:
        raise AssociationError(
            "선택한 변수들의 결측값을 제외한 뒤 "
            "분석 가능한 표본이 없습니다."
        )

    _validate_binary_target(
        analysis_df[
            TARGET_COLUMN
        ]
    )

    exposure_type: VariableType = (
        ANALYSIS_VARIABLE_TYPES[
            request.exposure
        ]
    )

    unadjusted = (
        _unadjusted_summary(
            analysis_df,
            request,
            exposure_type,
        )
    )

    adjusted = (
        _fit_logistic_association(
            analysis_df,
            request,
            exposure_type,
        )
    )

    psm = (
        _run_optional_psm(
            analysis_df,
            request,
        )
    )

    return {
        "request": {
            "target": (
                TARGET_COLUMN
            ),
            "exposure": (
                request.exposure
            ),
            "controls": list(
                request.controls
            ),
            "include_psm": (
                request.include_psm
            ),
        },

        "analysis": {
            "exposure_type": (
                exposure_type
            ),
            "input_rows": int(
                len(df)
            ),
            "sample_size": int(
                len(
                    analysis_df
                )
            ),
            "rows_excluded_due_to_missing": int(
                len(df)
                - len(
                    analysis_df
                )
            ),
            "unadjusted": (
                unadjusted
            ),
            "adjusted": (
                adjusted
            ),
            "psm": (
                psm
            ),
        },

        "interpretation_note": (
            "조정 전 결과는 관측된 단순 연관성을, "
            "Logistic Regression 결과는 선택한 통제변수를 "
            "고려한 조건부 연관성을 의미합니다. "
            "PSM을 수행한 경우에도 관측된 통제변수만 조정할 수 있으므로 "
            "어떤 결과도 확정적인 인과효과를 의미하지 않습니다."
        ),
    }