"""Adult Income 연관성 분석을 위한 범용 통계 도구.

이 모듈은 특정 관심 변수를 고정하지 않는다.

주요 역할:
    1. 이진 관심 변수와 이진 결과변수의 조정 전 집단 비교
    2. 이진 관심 변수에 대한 1:1 성향점수매칭(PSM)
    3. 매칭 전후 선택된 통제변수의 SMD 균형 진단

association.py가 전체 분석 흐름을 담당하고,
statistics.py는 필요한 통계 계산만 수행한다.

주의:
    PSM은 사용자가 선택한 관측 통제변수의 분포 차이를 줄이는 방법이다.
    관측되지 않은 교란요인을 제거하지 못하므로 결과를 확정적인
    인과효과로 해석하지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import (
    fisher_exact,
    norm,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from statsmodels.stats.contingency_tables import (
    mcnemar,
)

from src.config import (
    ANALYSIS_VARIABLE_TYPES,
    RANDOM_STATE,
    TARGET_COLUMN,
)


BALANCE_THRESHOLD = 0.1
MATCH_CANDIDATE_COUNT = 500

_INTERNAL_EXPOSURE_COLUMN = (
    "__exposure_binary__"
)


class StatisticsError(ValueError):
    """통계 분석 입력이나 계산 과정에서 발생하는 오류."""


# ============================================================
# 공통 검증
# ============================================================

def _validate_columns(
    df: pd.DataFrame,
    required: list[str],
    analysis_name: str,
) -> None:
    """분석에 필요한 열이 모두 존재하는지 확인한다."""

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise StatisticsError(
            f"{analysis_name} 입력은 "
            "pandas.DataFrame이어야 합니다."
        )

    if df.empty:
        raise StatisticsError(
            f"{analysis_name} 입력 데이터가 비어 있습니다."
        )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise StatisticsError(
            f"{analysis_name}에 필요한 열이 없습니다: "
            f"{missing}"
        )


def _validate_binary_outcome(series: pd.Series,) -> None:
    """결과변수가 정확히 0과 1을 포함하는지 확인한다."""

    try:
        numeric = pd.to_numeric(
            series.dropna(),
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise StatisticsError(
            f"결과 변수 '{series.name}'를 "
            "0/1 숫자로 해석할 수 없습니다."
        ) from exc

    values = set(numeric.unique())

    if values != {0, 1}:
        raise StatisticsError(
            f"결과 변수 '{series.name}'는 "
            "0과 1을 모두 포함해야 합니다. "
            f"현재 값: {values}"
        )


def _ordered_levels(series: pd.Series,) -> list:
    """이진 관심 변수의 기준·비교 범주 순서를 결정한다."""

    levels = (
        series
        .dropna()
        .unique()
        .tolist()
    )

    if len(levels) != 2:
        raise StatisticsError(
            f"관심 변수 '{series.name}'는 "
            "PSM 또는 이진 집단 비교를 위해 "
            "정확히 두 범주를 가져야 합니다. "
            f"현재 범주 수: {len(levels)}"
        )

    return sorted(levels,key=lambda value: str(value),)

def _encode_binary_exposure(series: pd.Series,) -> tuple[pd.Series, dict]:
    """두 범주의 관심 변수를 내부적으로 0/1로 변환한다.

    원래 범주값은 metadata에 보존한다.
    """

    levels = _ordered_levels(series)

    reference = levels[0]
    comparison = levels[1]

    encoded = (
        series
        .map(
            {
                reference: 0,
                comparison: 1,
            }
        )
        .astype("Int8")
    )

    metadata = {
        "reference_level": reference,
        "comparison_level": comparison,
    }

    return (
        encoded,
        metadata,
    )


def _format_p_value(p_value: float,) -> str:
    """화면 출력용 p-value 문자열을 생성한다."""

    if p_value == 0:
        return "< 1e-300"

    return f"{p_value:.6g}"


# ============================================================
# 이진 관심 변수 조정 전 비교
# ============================================================

def _odds_ratio_with_ci(
    comparison: pd.Series,
    reference: pd.Series,
    confidence: float = 0.95,
) -> tuple[
    float,
    float,
    float,
]:
    """두 집단의 odds ratio와 근사 신뢰구간을 계산한다.

    2×2 표에 빈 셀이 있으면 Haldane-Anscombe 보정으로
    네 셀 모두에 0.5를 더한다.
    """

    comparison_high = float(comparison.sum())
    comparison_low = float(len(comparison) - comparison_high)
    reference_high = float(reference.sum())
    reference_low = float(len(reference) - reference_high)

    cells = np.array(
        [
            comparison_high,
            comparison_low,
            reference_high,
            reference_low,
        ],
        dtype=float,
    )

    if (cells == 0).any():
        cells = cells + 0.5

    (
        comparison_high,
        comparison_low,
        reference_high,
        reference_low,
    ) = cells

    odds_ratio = float(
        (
            comparison_high
            * reference_low
        )
        / (
            comparison_low
            * reference_high
        )
    )

    standard_error = float(
        np.sqrt(
            1 / comparison_high
            + 1 / comparison_low
            + 1 / reference_high
            + 1 / reference_low
        )
    )

    alpha = 1 - confidence

    critical = float(
        norm.ppf(
            1 - alpha / 2
        )
    )

    log_odds_ratio = float(
        np.log(
            odds_ratio
        )
    )

    ci_low = float(
        np.exp(
            log_odds_ratio
            - critical * standard_error
        )
    )

    ci_high = float(
        np.exp(
            log_odds_ratio
            + critical * standard_error
        )
    )

    return (
        odds_ratio,
        ci_low,
        ci_high,
    )


def binary_group_association(
    df: pd.DataFrame,
    exposure: str,
    outcome: str = TARGET_COLUMN,
) -> dict:
    """이진 관심 변수의 조정 전 고소득 연관성을 계산한다.

    Returns:
        기준·비교 집단의 표본 수와 고소득률,
        비율 차이, risk ratio, odds ratio와 95% CI,
        Fisher exact p-value, Cohen's h를 반환한다.
    """

    _validate_columns(
        df,
        [
            exposure,
            outcome,
        ],
        "이진 집단 비교",
    )

    analysis = (
        df[
            [
                exposure,
                outcome,
            ]
        ]
        .dropna()
        .copy()
        .reset_index(drop=True)
    )

    if analysis.empty:
        raise StatisticsError(
            "결측값을 제외한 뒤 "
            "이진 집단 비교에 사용할 표본이 없습니다."
        )

    _validate_binary_outcome(
        analysis[outcome]
    )

    (
        exposure_binary,
        exposure_metadata,
    ) = _encode_binary_exposure(
        analysis[exposure]
    )

    analysis[
        _INTERNAL_EXPOSURE_COLUMN
    ] = exposure_binary.astype(int)

    analysis[outcome] = (
        pd.to_numeric(
            analysis[outcome],
            errors="raise",
        )
        .astype(int)
    )

    comparison = (
        analysis.loc[
            analysis[
                _INTERNAL_EXPOSURE_COLUMN
            ]
            == 1,
            outcome,
        ]
        .astype(float)
    )

    reference = (
        analysis.loc[
            analysis[
                _INTERNAL_EXPOSURE_COLUMN
            ]
            == 0,
            outcome,
        ]
        .astype(float)
    )

    if (
        len(comparison) < 1
        or len(reference) < 1
    ):
        raise StatisticsError(
            "두 관심 변수 집단 모두에 "
            "분석 가능한 표본이 필요합니다."
        )

    comparison_rate = float(
        comparison.mean()
    )

    reference_rate = float(
        reference.mean()
    )

    rate_difference = float(
        comparison_rate
        - reference_rate
    )

    risk_ratio = (
        float(
            comparison_rate
            / reference_rate
        )
        if reference_rate > 0
        else None
    )

    (
        odds_ratio,
        odds_ratio_ci_low,
        odds_ratio_ci_high,
    ) = _odds_ratio_with_ci(
        comparison,
        reference,
    )

    comparison_high = int(
        comparison.sum()
    )

    comparison_low = int(
        len(comparison)
        - comparison_high
    )

    reference_high = int(
        reference.sum()
    )

    reference_low = int(
        len(reference)
        - reference_high
    )

    contingency = np.array(
        [
            [
                comparison_high,
                comparison_low,
            ],
            [
                reference_high,
                reference_low,
            ],
        ]
    )

    _, p_value = fisher_exact(
        contingency,
        alternative="two-sided",
    )

    cohens_h = float(
        2
        * np.arcsin(
            np.sqrt(
                comparison_rate
            )
        )
        - 2
        * np.arcsin(
            np.sqrt(
                reference_rate
            )
        )
    )

    return {
        "request": {
            "exposure": exposure,
            "outcome": outcome,
        },
        "analysis": {
            "method": (
                "binary_group_association"
            ),
            "exposure_metadata": (
                exposure_metadata
            ),
            "reference_n": int(
                len(reference)
            ),
            "comparison_n": int(
                len(comparison)
            ),
            "reference_rate": (
                reference_rate
            ),
            "comparison_rate": (
                comparison_rate
            ),
            "rate_difference": (
                rate_difference
            ),
            "risk_ratio": (
                risk_ratio
            ),
            "odds_ratio": (
                odds_ratio
            ),
            "odds_ratio_ci_95_low": (
                odds_ratio_ci_low
            ),
            "odds_ratio_ci_95_high": (
                odds_ratio_ci_high
            ),
            "cohens_h": (
                cohens_h
            ),
            "fisher_exact_p_value": float(
                p_value
            ),
            "p_value_display": (
                _format_p_value(
                    float(
                        p_value
                    )
                )
            ),
        },
        "interpretation_note": (
            "이 결과는 통제변수를 적용하지 않은 "
            "조정 전 집단 비교이며 인과효과를 의미하지 않습니다."
        ),
    }


# ============================================================
# PSM 공변량 처리
# ============================================================

def _split_covariates(
    covariates: list[str],
) -> tuple[
    list[str],
    list[str],
]:
    """config.py의 변수 메타데이터로 PSM 공변량 유형을 구분한다."""

    numeric_covariates: list[str] = []
    categorical_covariates: list[str] = []

    unknown = [
        column
        for column in covariates
        if column not in ANALYSIS_VARIABLE_TYPES
    ]

    if unknown:
        raise StatisticsError(
            "PSM에서 지원하지 않는 통제 변수가 있습니다: "
            f"{unknown}"
        )

    for column in covariates:
        variable_type = (
            ANALYSIS_VARIABLE_TYPES[
                column
            ]
        )

        if variable_type == "continuous":
            numeric_covariates.append(
                column
            )

        elif variable_type in {
            "binary",
            "categorical",
        }:
            categorical_covariates.append(
                column
            )

        else:
            raise StatisticsError(
                f"변수 '{column}'의 분석 타입 "
                f"'{variable_type}'을 처리할 수 없습니다."
            )

    return (
        numeric_covariates,
        categorical_covariates,
    )


def _prepare_covariates(
    df: pd.DataFrame,
    numeric_covariates: list[str],
    categorical_covariates: list[str],
) -> pd.DataFrame:
    """PSM 전처리에 적합하도록 통제변수 dtype과 결측 표현을 정리한다."""

    result = df.copy()

    for column in numeric_covariates:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    # sklearn 버전에 따른 pd.NA 처리 차이를 피하기 위해
    # 범주형 결측값은 np.nan으로 통일한다.
    for column in categorical_covariates:
        result[column] = (
            result[column]
            .astype(object)
            .where(
                result[column].notna(),
                np.nan,
            )
        )

    return result


def _build_propensity_pipeline(
    numeric_covariates: list[str],
    categorical_covariates: list[str],
) -> Pipeline:
    """선택된 통제변수로 propensity score 모델을 구성한다."""

    transformers = []

    if numeric_covariates:
        numeric_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scale",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_covariates,
            )
        )

    if categorical_covariates:
        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="__MISSING__",
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_covariates,
            )
        )

    if not transformers:
        raise StatisticsError(
            "PSM에는 최소 1개의 통제 변수가 필요합니다."
        )

    preprocessing = ColumnTransformer(
        transformers
    )

    model = LogisticRegression(
        max_iter=2_000,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        [
            (
                "preprocessing",
                preprocessing,
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# SMD
# ============================================================

def _encode_covariates_for_smd(
    before: pd.DataFrame,
    after: pd.DataFrame,
    covariates: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """매칭 전후 데이터에 동일한 범주 인코딩을 적용한다."""

    combined = pd.concat(
        {
            "before": (
                before[
                    covariates
                ].copy()
            ),
            "after": (
                after[
                    covariates
                ].copy()
            ),
        },
        names=[
            "sample",
            "row",
        ],
    )

    categorical_columns = []

    for column in covariates:
        variable_type = (
            ANALYSIS_VARIABLE_TYPES[
                column
            ]
        )

        if variable_type == "continuous":
            combined[column] = pd.to_numeric(
                combined[column],
                errors="coerce",
            )

            # 수치형 결측 자체의 집단 차이도 진단한다.
            if combined[column].isna().any():
                combined[
                    f"{column}__missing"
                ] = (
                    combined[column]
                    .isna()
                    .astype(float)
                )

        else:
            combined[column] = (
                combined[column]
                .astype("string")
                .fillna(
                    "__MISSING__"
                )
            )

            categorical_columns.append(
                column
            )

    if categorical_columns:
        combined = pd.get_dummies(
            combined,
            columns=categorical_columns,
            drop_first=False,
            dtype=float,
        )

    # 모든 표본에서 값이 존재하지 않는 원래 수치 열은
    # SMD를 계산할 정보가 없으므로 제거한다.
    combined = combined.loc[
        :,
        ~combined.isna().all(axis=0),
    ]

    before_encoded = (
        combined
        .xs(
            "before",
            level="sample",
        )
        .reset_index(drop=True)
    )

    after_encoded = (
        combined
        .xs(
            "after",
            level="sample",
        )
        .reset_index(drop=True)
    )

    return (
        before_encoded,
        after_encoded,
    )


def _calculate_smd(
    encoded: pd.DataFrame,
    exposure_binary: pd.Series,
) -> pd.Series:
    """인코딩된 공변량의 절대 SMD를 계산한다."""

    exposure_values = (
        exposure_binary
        .reset_index(drop=True)
        .astype(int)
    )

    comparison = (
        encoded.loc[
            exposure_values == 1
        ]
    )

    reference = (
        encoded.loc[
            exposure_values == 0
        ]
    )

    if (
        len(comparison) < 2
        or len(reference) < 2
    ):
        raise StatisticsError(
            "SMD 계산에는 두 집단에 각각 "
            "최소 2개 표본이 필요합니다."
        )

    comparison_mean = (
        comparison.mean()
    )

    reference_mean = (
        reference.mean()
    )

    mean_difference = (
        comparison_mean
        - reference_mean
    )

    pooled_sd = np.sqrt(
        (
            comparison.var(
                ddof=1
            )
            + reference.var(
                ddof=1
            )
        )
        / 2
    )

    smd = (
        mean_difference.abs()
        / pooled_sd
    )

    zero_sd = (
        pooled_sd == 0
    )

    smd.loc[
        zero_sd
        & (
            mean_difference.abs()
            == 0
        )
    ] = 0.0

    smd.loc[
        zero_sd
        & (
            mean_difference.abs()
            > 0
        )
    ] = np.inf

    # 한 집단에서 값이 전혀 없어 분산을 정의할 수 없는 경우
    # 균형이 좋다고 오인하지 않도록 무한대로 처리한다.
    smd.loc[
        smd.isna()
    ] = np.inf

    return smd


def _smd_table(
    before: pd.DataFrame,
    after: pd.DataFrame,
    covariates: list[str],
) -> pd.DataFrame:
    """선택된 통제변수의 매칭 전후 절대 SMD를 계산한다."""

    (
        before_encoded,
        after_encoded,
    ) = _encode_covariates_for_smd(
        before,
        after,
        covariates,
    )

    smd_before = _calculate_smd(
        before_encoded,
        before[
            _INTERNAL_EXPOSURE_COLUMN
        ],
    )

    smd_after = _calculate_smd(
        after_encoded,
        after[
            _INTERNAL_EXPOSURE_COLUMN
        ],
    )

    return (
        pd.concat(
            [
                smd_before.rename(
                    "smd_before"
                ),
                smd_after.rename(
                    "smd_after"
                ),
            ],
            axis=1,
        )
        .rename_axis(
            "covariate"
        )
        .reset_index()
        .sort_values(
            "smd_after",
            ascending=False,
            ignore_index=True,
        )
    )


def _finite_or_none(
    value: float,
) -> float | None:
    """JSON/API 응답에 사용할 수 있도록 비유한값을 None으로 변환한다."""

    value = float(
        value
    )

    if not np.isfinite(
        value
    ):
        return None

    return value


# ============================================================
# Propensity Score Matching
# ============================================================

def propensity_score_matching(
    df: pd.DataFrame,
    exposure: str,
    covariates: list[str] | tuple[str, ...],
    outcome: str = TARGET_COLUMN,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    """이진 관심 변수에 대해 1:1 성향점수매칭을 수행한다.

    Args:
        df:
            정제된 Adult 데이터.

        exposure:
            고소득 여부와의 관계를 확인할 이진 관심 변수.

        covariates:
            propensity score 계산에 사용할 통제 변수.
            홈페이지에서 사용자가 선택한 controls를 전달한다.

        outcome:
            이진 결과변수. 기본값은 high_income.

    Returns:
        matched:
            1:1로 매칭된 표본.

        balance:
            선택된 통제변수의 매칭 전후 SMD.

        result:
            매칭 품질, 집단별 고소득률, McNemar 검정,
            균형 진단을 포함한 결과 dict.

    Notes:
        현재 구현은 두 범주의 관심 변수만 지원한다.
        연속형 또는 세 범주 이상의 관심 변수는 association.py의
        회귀 기반 연관성 분석을 사용한다.
    """

    covariates = list(
        covariates
    )

    if not covariates:
        raise StatisticsError(
            "PSM에는 최소 1개의 통제 변수가 필요합니다."
        )

    if len(
        set(covariates)
    ) != len(covariates):
        raise StatisticsError(
            "PSM 통제 변수에 중복된 변수가 있습니다."
        )

    if exposure in covariates:
        raise StatisticsError(
            "관심 변수는 PSM 통제 변수에 "
            "동시에 포함할 수 없습니다."
        )

    if outcome in covariates:
        raise StatisticsError(
            "결과 변수는 PSM 통제 변수로 "
            "사용할 수 없습니다."
        )

    required_columns = [
        exposure,
        outcome,
        *covariates,
    ]

    _validate_columns(
        df,
        required_columns,
        "성향점수매칭",
    )

    analysis = (
        df[
            required_columns
        ]
        .dropna(
            subset=[
                exposure,
                outcome,
            ]
        )
        .copy()
        .reset_index(drop=True)
    )

    if analysis.empty:
        raise StatisticsError(
            "성향점수매칭에 사용할 표본이 없습니다."
        )

    _validate_binary_outcome(
        analysis[outcome]
    )

    (
        exposure_binary,
        exposure_metadata,
    ) = _encode_binary_exposure(
        analysis[exposure]
    )

    analysis[
        _INTERNAL_EXPOSURE_COLUMN
    ] = (
        exposure_binary
        .astype(int)
    )

    analysis[outcome] = (
        pd.to_numeric(
            analysis[outcome],
            errors="raise",
        )
        .astype(int)
    )

    (
        numeric_covariates,
        categorical_covariates,
    ) = _split_covariates(
        covariates
    )

    analysis = _prepare_covariates(
        analysis,
        numeric_covariates,
        categorical_covariates,
    )

    comparison_group = (
        analysis.loc[
            analysis[
                _INTERNAL_EXPOSURE_COLUMN
            ]
            == 1
        ]
        .copy()
    )

    reference_group = (
        analysis.loc[
            analysis[
                _INTERNAL_EXPOSURE_COLUMN
            ]
            == 0
        ]
        .copy()
    )

    if (
        len(comparison_group) < 2
        or len(reference_group) < 2
    ):
        raise StatisticsError(
            "PSM에는 두 관심 변수 집단에 "
            "각각 최소 2개의 표본이 필요합니다."
        )

    propensity_model = (
        _build_propensity_pipeline(
            numeric_covariates,
            categorical_covariates,
        )
    )

    try:
        propensity_model.fit(
            analysis[covariates],
            analysis[
                _INTERNAL_EXPOSURE_COLUMN
            ],
        )

        analysis[
            "propensity_score"
        ] = (
            propensity_model
            .predict_proba(
                analysis[
                    covariates
                ]
            )[:, 1]
        )

    except Exception as exc:
        raise StatisticsError(
            "Propensity score 모델 적합에 실패했습니다: "
            f"{exc}"
        ) from exc

    comparison_group = (
        analysis.loc[
            analysis[
                _INTERNAL_EXPOSURE_COLUMN
            ]
            == 1
        ]
        .copy()
    )

    reference_group = (
        analysis.loc[
            analysis[
                _INTERNAL_EXPOSURE_COLUMN
            ]
            == 0
        ]
        .copy()
    )

    comparison_before_support = int(
        len(
            comparison_group
        )
    )

    reference_before_support = int(
        len(
            reference_group
        )
    )

    # 두 집단에서 실제로 관측되는 propensity score 범위만 사용한다.
    common_support_lower = max(
        float(
            comparison_group[
                "propensity_score"
            ].min()
        ),
        float(
            reference_group[
                "propensity_score"
            ].min()
        ),
    )

    common_support_upper = min(
        float(
            comparison_group[
                "propensity_score"
            ].max()
        ),
        float(
            reference_group[
                "propensity_score"
            ].max()
        ),
    )

    if (
        common_support_lower
        > common_support_upper
    ):
        raise StatisticsError(
            "두 집단의 propensity score 공통지지 영역이 없습니다."
        )

    comparison_group = (
        comparison_group.loc[
            comparison_group[
                "propensity_score"
            ].between(
                common_support_lower,
                common_support_upper,
            )
        ]
        .copy()
    )

    reference_group = (
        reference_group.loc[
            reference_group[
                "propensity_score"
            ].between(
                common_support_lower,
                common_support_upper,
            )
        ]
        .copy()
    )

    if (
        len(comparison_group) < 2
        or len(reference_group) < 2
    ):
        raise StatisticsError(
            "공통지지 영역을 적용한 뒤 "
            "PSM에 필요한 표본이 부족합니다."
        )

    clipped = (
        analysis[
            "propensity_score"
        ]
        .clip(
            1e-6,
            1 - 1e-6,
        )
    )

    analysis[
        "propensity_logit"
    ] = logit(
        clipped
    )

    comparison_group = (
        analysis.loc[
            comparison_group.index
        ]
        .copy()
    )

    reference_group = (
        analysis.loc[
            reference_group.index
        ]
        .copy()
    )

    propensity_logit_sd = float(
        analysis[
            "propensity_logit"
        ].std(
            ddof=1
        )
    )

    if not np.isfinite(
        propensity_logit_sd
    ):
        raise StatisticsError(
            "Propensity score의 logit 표준편차를 "
            "계산할 수 없습니다."
        )

    # 일반적인 0.2 × logit(PS) SD caliper를 사용한다.
    # 모든 propensity score가 사실상 동일한 경우에도
    # 수치 오차 때문에 매칭이 실패하지 않도록 최소 허용값을 둔다.
    caliper = max(
        0.2
        * propensity_logit_sd,
        1e-12,
    )

    neighbor_count = min(
        MATCH_CANDIDATE_COUNT,
        len(
            reference_group
        ),
    )

    neighbors = NearestNeighbors(
        n_neighbors=neighbor_count
    )

    neighbors.fit(
        reference_group[
            [
                "propensity_logit",
            ]
        ]
    )

    distances, indices = (
        neighbors.kneighbors(
            comparison_group[
                [
                    "propensity_logit",
                ]
            ]
        )
    )

    pairs: list[dict] = []

    used_reference_positions: set[int] = set()

    # 가장 가까운 후보와도 거리가 큰 표본부터 처리하면
    # 공통지지 경계의 비교집단 표본이 먼저 매칭 기회를 얻는다.
    comparison_order = (
        np.argsort(
            distances[:, 0]
        )[::-1]
    )

    pair_id = 0

    for comparison_position in (
        comparison_order
    ):
        selected_reference_position = (
            None
        )

        selected_distance = (
            None
        )

        for (
            distance,
            reference_position,
        ) in zip(
            distances[
                comparison_position
            ],
            indices[
                comparison_position
            ],
        ):
            reference_position = int(
                reference_position
            )

            if distance > caliper:
                break

            if (
                reference_position
                not in used_reference_positions
            ):
                selected_reference_position = (
                    reference_position
                )

                selected_distance = float(
                    distance
                )

                break

        if (
            selected_reference_position
            is None
        ):
            continue

        used_reference_positions.add(
            selected_reference_position
        )

        comparison_row = (
            comparison_group.iloc[
                int(
                    comparison_position
                )
            ]
        )

        reference_row = (
            reference_group.iloc[
                selected_reference_position
            ]
        )

        pairs.extend(
            [
                {
                    **comparison_row.to_dict(),
                    "source_index": (
                        comparison_row.name
                    ),
                    "pair_id": pair_id,
                    "matched_role": (
                        "comparison"
                    ),
                    "match_distance": (
                        selected_distance
                    ),
                },
                {
                    **reference_row.to_dict(),
                    "source_index": (
                        reference_row.name
                    ),
                    "pair_id": pair_id,
                    "matched_role": (
                        "reference"
                    ),
                    "match_distance": (
                        selected_distance
                    ),
                },
            ]
        )

        pair_id += 1

    matched = pd.DataFrame(
        pairs
    )

    matched_pairs = pair_id

    if matched_pairs < 2:
        raise StatisticsError(
            "caliper 안에서 충분한 매칭쌍을 "
            "생성하지 못했습니다."
        )

    paired_outcomes = (
        matched
        .pivot(
            index="pair_id",
            columns="matched_role",
            values=outcome,
        )
        .dropna()
    )

    paired_outcomes[
        "comparison"
    ] = (
        paired_outcomes[
            "comparison"
        ]
        .astype(int)
    )

    paired_outcomes[
        "reference"
    ] = (
        paired_outcomes[
            "reference"
        ]
        .astype(int)
    )

    matched_table = (
        pd.crosstab(
            paired_outcomes[
                "comparison"
            ],
            paired_outcomes[
                "reference"
            ],
        )
        .reindex(
            index=[
                0,
                1,
            ],
            columns=[
                0,
                1,
            ],
            fill_value=0,
        )
    )

    mcnemar_result = mcnemar(
        matched_table.to_numpy(),
        exact=True,
    )

    comparison_outcome = (
        matched.loc[
            matched[
                "matched_role"
            ]
            == "comparison",
            outcome,
        ]
        .astype(float)
    )

    reference_outcome = (
        matched.loc[
            matched[
                "matched_role"
            ]
            == "reference",
            outcome,
        ]
        .astype(float)
    )

    comparison_rate = float(
        comparison_outcome.mean()
    )

    reference_rate = float(
        reference_outcome.mean()
    )

    rate_difference = float(
        comparison_rate
        - reference_rate
    )

    balance = _smd_table(
        analysis,
        matched,
        covariates,
    )

    max_smd_before_raw = float(
        balance[
            "smd_before"
        ].max()
    )

    max_smd_after_raw = float(
        balance[
            "smd_after"
        ].max()
    )

    balanced = bool(
        np.isfinite(
            balance[
                "smd_after"
            ]
        ).all()
        and (
            balance[
                "smd_after"
            ]
            < BALANCE_THRESHOLD
        ).all()
    )

    result = {
        "request": {
            "exposure": exposure,
            "outcome": outcome,
            "covariates": (
                covariates
            ),
        },

        "exposure_metadata": (
            exposure_metadata
        ),

        "matching": {
            "method": (
                "1:1 greedy nearest-neighbor "
                "propensity-score matching "
                "without replacement"
            ),
            "matched_pairs": int(
                matched_pairs
            ),
            "comparison_before_common_support": (
                comparison_before_support
            ),
            "reference_before_common_support": (
                reference_before_support
            ),
            "comparison_in_common_support": int(
                len(
                    comparison_group
                )
            ),
            "reference_in_common_support": int(
                len(
                    reference_group
                )
            ),
            "comparison_retention_rate": float(
                matched_pairs
                / comparison_before_support
            ),
            "common_support_lower": (
                common_support_lower
            ),
            "common_support_upper": (
                common_support_upper
            ),
            "caliper": float(
                caliper
            ),
            "mean_match_distance": float(
                matched.loc[
                    matched[
                        "matched_role"
                    ]
                    == "comparison",
                    "match_distance",
                ].mean()
            ),
        },

        "outcome_comparison": {
            "reference_rate": (
                reference_rate
            ),
            "comparison_rate": (
                comparison_rate
            ),
            "matched_rate_difference": (
                rate_difference
            ),
            "mcnemar_statistic": float(
                mcnemar_result.statistic
            ),
            "mcnemar_p_value": float(
                mcnemar_result.pvalue
            ),
            "p_value_display": (
                _format_p_value(
                    float(
                        mcnemar_result.pvalue
                    )
                )
            ),
            "comparison_only_positive_pairs": int(
                matched_table.loc[
                    1,
                    0,
                ]
            ),
            "reference_only_positive_pairs": int(
                matched_table.loc[
                    0,
                    1,
                ]
            ),
        },

        "balance": {
            "threshold": (
                BALANCE_THRESHOLD
            ),
            "max_smd_before": (
                _finite_or_none(
                    max_smd_before_raw
                )
            ),
            "max_smd_after": (
                _finite_or_none(
                    max_smd_after_raw
                )
            ),
            "imbalanced_covariates_before": int(
                (
                    balance[
                        "smd_before"
                    ]
                    >= BALANCE_THRESHOLD
                ).sum()
            ),
            "imbalanced_covariates_after": int(
                (
                    balance[
                        "smd_after"
                    ]
                    >= BALANCE_THRESHOLD
                ).sum()
            ),
            "balanced_under_threshold": (
                balanced
            ),
        },

        "interpretation_note": (
            "PSM 결과는 사용자가 선택한 관측 통제변수의 "
            "분포 차이를 줄인 뒤 비교한 연관성입니다. "
            "관측되지 않은 교란요인을 통제하지 못하므로 "
            "확정적인 인과효과로 해석하지 않습니다."
        ),
    }

    return (
        matched,
        balance,
        result,
    )