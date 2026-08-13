"""Adult Income 웹 서비스의 사용자용 시각화.

이 모듈은 분석 결과를 다시 계산하거나 파일에서 읽지 않는다.

association.py와 modeling.py가 반환한 결과 객체와
필요한 DataFrame을 받아 Plotly Figure로 변환한다.

주요 역할:
    1. 관심 변수와 high_income의 조정 전 관계 시각화
    2. Logistic Regression의 조정된 Odds Ratio 시각화
    3. PSM 수행 시 매칭 전후 공변량 균형 시각화
    4. 사용자 입력에 대한 고소득 예측 확률 시각화

모든 함수는 파일을 저장하지 않고 Figure 객체를 반환한다.
웹 UI는 반환된 Figure를 직접 표시한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class VisualizationError(ValueError):
    """시각화 입력이나 결과 구조가 올바르지 않을 때 발생하는 오류."""


# ============================================================
# 공통 검증
# ============================================================

def _require_columns(
    df: pd.DataFrame,
    columns: list[str],
    chart_name: str,
) -> None:
    """그래프 생성에 필요한 DataFrame 열을 검증한다."""

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise VisualizationError(
            f"{chart_name} 입력은 "
            "pandas.DataFrame이어야 합니다."
        )

    missing_columns = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing_columns:
        raise VisualizationError(
            f"{chart_name}에 필요한 열이 없습니다: "
            f"{missing_columns}"
        )


def _validate_association_result(
    result: dict,
) -> None:
    """association.py 결과의 최소 구조를 확인한다."""

    if not isinstance(
        result,
        dict,
    ):
        raise VisualizationError(
            "연관성 분석 결과는 dict여야 합니다."
        )

    if "request" not in result:
        raise VisualizationError(
            "연관성 분석 결과에 request가 없습니다."
        )

    if "analysis" not in result:
        raise VisualizationError(
            "연관성 분석 결과에 analysis가 없습니다."
        )

    request = result["request"]
    analysis = result["analysis"]

    required_request_keys = {
        "target",
        "exposure",
        "controls",
    }

    missing_request_keys = (
        required_request_keys
        - set(request)
    )

    if missing_request_keys:
        raise VisualizationError(
            "연관성 분석 request에 필요한 값이 없습니다: "
            f"{sorted(missing_request_keys)}"
        )

    required_analysis_keys = {
        "exposure_type",
        "unadjusted",
        "adjusted",
    }

    missing_analysis_keys = (
        required_analysis_keys
        - set(analysis)
    )

    if missing_analysis_keys:
        raise VisualizationError(
            "연관성 분석 결과에 필요한 값이 없습니다: "
            f"{sorted(missing_analysis_keys)}"
        )


# ============================================================
# 조정 전 연관성 시각화
# ============================================================

def plot_unadjusted_association(
    result: dict,
) -> go.Figure:
    """association.py가 계산한 동일 표본의 조정 전 결과를 시각화한다."""

    _validate_association_result(
        result
    )

    request = result["request"]
    analysis = result["analysis"]

    exposure = request[
        "exposure"
    ]

    exposure_type = analysis[
        "exposure_type"
    ]

    unadjusted = analysis[
        "unadjusted"
    ]

    # --------------------------------------------------------
    # 이진 관심 변수
    # --------------------------------------------------------

    if exposure_type == "binary":
        metadata = unadjusted[
            "exposure_metadata"
        ]

        chart_data = pd.DataFrame(
            {
                "level": [
                    metadata[
                        "reference_level"
                    ],
                    metadata[
                        "comparison_level"
                    ],
                ],
                "sample_size": [
                    unadjusted[
                        "reference_n"
                    ],
                    unadjusted[
                        "comparison_n"
                    ],
                ],
                "high_income_rate_percent": [
                    (
                        unadjusted[
                            "reference_rate"
                        ]
                        * 100
                    ),
                    (
                        unadjusted[
                            "comparison_rate"
                        ]
                        * 100
                    ),
                ],
            }
        )

        figure = px.bar(
            chart_data,
            x="level",
            y="high_income_rate_percent",
            custom_data=[
                "sample_size",
            ],
            title=(
                f"{exposure}별 조정 전 고소득률"
            ),
            labels={
                "level": exposure,
                "high_income_rate_percent": (
                    "고소득률 (%)"
                ),
            },
        )

    # --------------------------------------------------------
    # 다범주형 관심 변수
    # --------------------------------------------------------

    elif exposure_type == "categorical":
        chart_data = pd.DataFrame(
            unadjusted[
                "groups"
            ]
        )

        chart_data[
            "high_income_rate_percent"
        ] = (
            chart_data[
                "target_rate"
            ]
            * 100
        )

        chart_data = (
            chart_data
            .sort_values(
                "high_income_rate_percent",
                ascending=True,
            )
        )

        figure = px.bar(
            chart_data,
            x=exposure,
            y="high_income_rate_percent",
            custom_data=[
                "n",
            ],
            title=(
                f"{exposure}별 조정 전 고소득률"
            ),
            labels={
                exposure: exposure,
                "high_income_rate_percent": (
                    "고소득률 (%)"
                ),
            },
        )

    # --------------------------------------------------------
    # 연속형 관심 변수
    # --------------------------------------------------------

    elif exposure_type == "continuous":
        chart_data = pd.DataFrame(
            unadjusted[
                "bins"
            ]
        )

        chart_data[
            "high_income_rate_percent"
        ] = (
            chart_data[
                "target_rate"
            ]
            * 100
        )

        figure = px.line(
            chart_data,
            x="exposure_mean",
            y="high_income_rate_percent",
            markers=True,
            custom_data=[
                "exposure_min",
                "exposure_max",
                "n",
            ],
            title=(
                f"{exposure}와 조정 전 고소득률"
            ),
            labels={
                "exposure_mean": (
                    f"{exposure} 구간 평균"
                ),
                "high_income_rate_percent": (
                    "고소득률 (%)"
                ),
            },
        )

        figure.update_traces(
            hovertemplate=(
                f"{exposure} 평균: %{{x:.2f}}"
                "<br>구간: "
                "%{customdata[0]:.2f}"
                " ~ "
                "%{customdata[1]:.2f}"
                "<br>고소득률: %{y:.2f}%"
                "<br>표본 수: %{customdata[2]:,}"
                "<extra></extra>"
            )
        )

        return figure

    else:
        raise VisualizationError(
            "지원하지 않는 관심 변수 유형입니다: "
            f"{exposure_type}"
        )

    figure.update_traces(
        hovertemplate=(
            "%{x}"
            "<br>고소득률: %{y:.2f}%"
            "<br>표본 수: %{customdata[0]:,}"
            "<extra></extra>"
        )
    )

    figure.update_yaxes(
        rangemode="tozero"
    )

    return figure


# ============================================================
# 조정 후 Odds Ratio 시각화
# ============================================================

def _exposure_effect_label(
    term: str,
    exposure: str,
    exposure_type: str,
    metadata: dict,
) -> str:
    """회귀계수 이름을 사용자에게 보여줄 라벨로 변환한다."""

    if exposure_type == "continuous":
        return (
            f"{exposure} "
            "(1단위 증가)"
        )

    if exposure_type == "binary":
        reference = (
            metadata.get(
                "reference_level"
            )
        )

        comparison = (
            metadata.get(
                "comparison_level"
            )
        )

        if (
            reference is not None
            and comparison is not None
        ):
            return (
                f"{comparison} vs "
                f"{reference}"
            )

        return exposure

    reference = metadata.get(
        "reference_level"
    )

    prefix = (
        f"{exposure}_"
    )

    level = (
        term[len(prefix):]
        if term.startswith(prefix)
        else term
    )

    if reference is None:
        return str(
            level
        )

    return (
        f"{level} vs {reference}"
    )


def plot_adjusted_association(
    result: dict,
) -> go.Figure:
    """관심 변수의 조정된 Odds Ratio와 95% CI를 forest plot으로 표시한다."""

    _validate_association_result(
        result
    )

    request = result[
        "request"
    ]

    analysis_result = result[
        "analysis"
    ]

    adjusted = analysis_result[
        "adjusted"
    ]

    exposure = request[
        "exposure"
    ]

    exposure_type = (
        analysis_result[
            "exposure_type"
        ]
    )

    exposure_effects = (
        adjusted.get(
            "exposure_effects",
            []
        )
    )

    if not exposure_effects:
        raise VisualizationError(
            "조정된 관심 변수 효과가 없습니다."
        )

    metadata = (
        adjusted.get(
            "exposure_metadata",
            {}
        )
    )

    rows: list[dict] = []

    for effect in exposure_effects:
        if not effect.get(
            "estimable",
            True,
        ):
            continue

        if (
            effect.get(
                "odds_ratio"
            )
            is None
            or effect.get(
                "ci_95_low"
            )
            is None
            or effect.get(
                "ci_95_high"
            )
            is None
        ):
            continue
        
        odds_ratio = float(
            effect[
                "odds_ratio"
            ]
        )

        ci_low = float(
            effect[
                "ci_95_low"
            ]
        )

        ci_high = float(
            effect[
                "ci_95_high"
            ]
        )

        if (
            odds_ratio <= 0
            or ci_low <= 0
            or ci_high <= 0
        ):
            raise VisualizationError(
                "Odds Ratio와 신뢰구간은 "
                "양수여야 합니다."
            )

        rows.append(
            {
                "label": (
                    _exposure_effect_label(
                        term=str(
                            effect[
                                "term"
                            ]
                        ),
                        exposure=exposure,
                        exposure_type=(
                            exposure_type
                        ),
                        metadata=metadata,
                    )
                ),
                "odds_ratio": (
                    odds_ratio
                ),
                "ci_low": (
                    ci_low
                ),
                "ci_high": (
                    ci_high
                ),
                "p_value": float(
                    effect[
                        "p_value"
                    ]
                ),
            }
        )

    if not rows:
        raise VisualizationError(
            "관심 변수의 Odds Ratio를 안정적으로 "
            "추정할 수 있는 범주가 없습니다."
        )

    chart_data = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "odds_ratio",
            ascending=True,
        )
    )

    error_plus = (
        chart_data[
            "ci_high"
        ]
        - chart_data[
            "odds_ratio"
        ]
    )

    error_minus = (
        chart_data[
            "odds_ratio"
        ]
        - chart_data[
            "ci_low"
        ]
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=chart_data[
                "odds_ratio"
            ],
            y=chart_data[
                "label"
            ],
            mode="markers",
            customdata=np.column_stack(
                [
                    chart_data[
                        "ci_low"
                    ],
                    chart_data[
                        "ci_high"
                    ],
                    chart_data[
                        "p_value"
                    ],
                ]
            ),
            error_x={
                "type": "data",
                "symmetric": False,
                "array": error_plus,
                "arrayminus": error_minus,
            },
            hovertemplate=(
                "%{y}"
                "<br>Adjusted OR: %{x:.3f}"
                "<br>95% CI: "
                "%{customdata[0]:.3f}"
                " – "
                "%{customdata[1]:.3f}"
                "<br>p-value: "
                "%{customdata[2]:.4g}"
                "<extra></extra>"
            ),
            name="Adjusted OR",
        )
    )

    figure.add_vline(
        x=1,
        line_dash="dash",
        annotation_text="OR = 1",
    )

    figure.update_layout(
        title=(
            f"{exposure}의 조정된 Odds Ratio"
        ),
        xaxis_title=(
            "Adjusted Odds Ratio "
            "(log scale)"
        ),
        yaxis_title="",
        showlegend=False,
    )

    figure.update_xaxes(
        type="log"
    )

    return figure


# ============================================================
# PSM 균형 시각화
# ============================================================

def plot_psm_balance(
    balance: pd.DataFrame,
    *,
    threshold: float = 0.1,
    top_n: int = 20,
) -> go.Figure:
    """PSM의 매칭 전후 SMD를 Love Plot 형태로 표시한다.

    statistics.py에서 반환한 balance DataFrame을 직접 사용하며,
    CSV 파일을 다시 읽지 않는다.
    """

    _require_columns(
        balance,
        [
            "covariate",
            "smd_before",
            "smd_after",
        ],
        "PSM 균형 시각화",
    )

    if top_n < 1:
        raise VisualizationError(
            "top_n은 1 이상이어야 합니다."
        )

    chart_data = (
        balance.copy()
    )

    chart_data[
        "max_smd"
    ] = (
        chart_data[
            [
                "smd_before",
                "smd_after",
            ]
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .abs()
        .max(axis=1)
    )

    chart_data = (
        chart_data
        .sort_values(
            "max_smd",
            ascending=False,
        )
        .head(top_n)
        .sort_values(
            "max_smd",
            ascending=True,
        )
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=chart_data[
                "smd_before"
            ],
            y=chart_data[
                "covariate"
            ],
            mode="markers",
            name="매칭 전",
            hovertemplate=(
                "%{y}"
                "<br>매칭 전 SMD: %{x:.3f}"
                "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=chart_data[
                "smd_after"
            ],
            y=chart_data[
                "covariate"
            ],
            mode="markers",
            name="매칭 후",
            hovertemplate=(
                "%{y}"
                "<br>매칭 후 SMD: %{x:.3f}"
                "<extra></extra>"
            ),
        )
    )

    figure.add_vline(
        x=threshold,
        line_dash="dash",
        annotation_text=(
            f"SMD = {threshold}"
        ),
    )

    figure.update_layout(
        title=(
            "PSM 매칭 전후 통제변수 균형"
        ),
        xaxis_title=(
            "Absolute Standardized "
            "Mean Difference"
        ),
        yaxis_title="",
    )

    return figure


# ============================================================
# 고소득 예측 확률 시각화
# ============================================================

def plot_prediction_probability(
    result: dict,
) -> go.Figure:
    """한 입력의 고소득 예측 확률을 사용자용 그래프로 표시한다."""

    if not isinstance(
        result,
        dict,
    ):
        raise VisualizationError(
            "예측 결과는 dict여야 합니다."
        )

    prediction = result.get(
        "prediction"
    )

    if not isinstance(
        prediction,
        dict,
    ):
        raise VisualizationError(
            "예측 결과에 prediction 정보가 없습니다."
        )

    if (
        "high_income_probability"
        not in prediction
    ):
        raise VisualizationError(
            "예측 결과에 고소득 확률이 없습니다."
        )

    probability = float(
        prediction[
            "high_income_probability"
        ]
    )

    if not (
        0 <= probability <= 1
    ):
        raise VisualizationError(
            "고소득 예측 확률은 "
            "0과 1 사이여야 합니다."
        )

    chart_data = pd.DataFrame(
        {
            "income_class": [
                "<=50K",
                ">50K",
            ],
            "probability_percent": [
                (
                    1
                    - probability
                )
                * 100,
                probability
                * 100,
            ],
        }
    )

    figure = px.bar(
        chart_data,
        x="probability_percent",
        y="income_class",
        orientation="h",
        text="probability_percent",
        title="입력 조건의 고소득 예측 확률",
        labels={
            "income_class": (
                "소득 클래스"
            ),
            "probability_percent": (
                "모델 예측 확률 (%)"
            ),
        },
    )

    figure.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate=(
            "%{y}"
            "<br>예측 확률: %{x:.2f}%"
            "<extra></extra>"
        ),
    )

    figure.update_xaxes(
        range=[
            0,
            100,
        ]
    )

    figure.update_layout(
        showlegend=False
    )

    return figure


# ============================================================
# 연관성 분석 시각화 묶음
# ============================================================

def create_association_visualizations(
    result: dict,
) -> dict[str, go.Figure]:
    """한 연관성 분석 결과에 필요한 사용자용 Figure를 생성한다."""

    figures = {
        "unadjusted": (
            plot_unadjusted_association(
                result
            )
        ),
        "adjusted": (
            plot_adjusted_association(
                result
            )
        ),
    }

    psm = (
        result[
            "analysis"
        ].get(
            "psm"
        )
    )

    if psm is not None:
        balance = pd.DataFrame(
            psm[
                "balance"
            ]
        )

        figures[
            "psm_balance"
        ] = (
            plot_psm_balance(
                balance
            )
        )

    return figures

def plot_prediction_explanation(
    result: dict,
    *,
    top_n: int = 8,
) -> go.Figure:
    """현재 입력값을 대표값으로 바꿨을 때의 예측 확률 차이를 표시한다."""

    explanation = (
        result.get(
            "explanation",
            {}
        )
    )

    features = explanation.get(
        "features"
    )

    if not features:
        raise VisualizationError(
            "개인 예측 설명 데이터가 없습니다."
        )

    chart_data = (
        pd.DataFrame(
            features
        )
        .head(
            top_n
        )
        .sort_values(
            "impact_percentage_points",
            ascending=True,
        )
    )

    figure = px.bar(
        chart_data,
        x="impact_percentage_points",
        y="feature",
        orientation="h",
        custom_data=[
            "current_value",
            "reference_value",
        ],
        title=(
            "현재 입력값에 따른 모델 예측 변화"
        ),
        labels={
            "feature": "변수",
            "impact_percentage_points": (
                "대표값 대비 예측 확률 차이 (%p)"
            ),
        },
    )

    figure.add_vline(
        x=0,
        line_dash="dash",
    )

    figure.update_traces(
        hovertemplate=(
            "%{y}"
            "<br>확률 차이: %{x:.2f}%p"
            "<br>현재 값: %{customdata[0]}"
            "<br>대표값: %{customdata[1]}"
            "<extra></extra>"
        )
    )

    return figure

def plot_what_if_simulation(
    what_if: pd.DataFrame,
) -> go.Figure:
    """한 변수의 값만 변경했을 때 모델 예측 확률의 변화를 표시한다."""

    _require_columns(
        what_if,
        [
            "feature",
            "value",
            "high_income_probability_percent",
        ],
        "What-if 시각화",
    )

    if what_if.empty:
        raise VisualizationError(
            "What-if 결과가 비어 있습니다."
        )

    feature = str(
        what_if[
            "feature"
        ].iloc[0]
    )

    numeric_values = pd.to_numeric(
        what_if[
            "value"
        ],
        errors="coerce",
    )

    # 모든 값이 숫자로 해석되면 연속적인 변화로 표시한다.
    if numeric_values.notna().all():
        chart_data = (
            what_if.copy()
        )

        chart_data[
            "value"
        ] = numeric_values

        chart_data = (
            chart_data
            .sort_values(
                "value"
            )
        )

        figure = px.line(
            chart_data,
            x="value",
            y=(
                "high_income_probability_percent"
            ),
            markers=True,
            title=(
                f"{feature} 변화에 따른 모델 예측 확률"
            ),
            labels={
                "value": feature,
                "high_income_probability_percent": (
                    "고소득 예측 확률 (%)"
                ),
            },
        )

    else:
        figure = px.bar(
            what_if,
            x="value",
            y=(
                "high_income_probability_percent"
            ),
            title=(
                f"{feature} 변화에 따른 모델 예측 확률"
            ),
            labels={
                "value": feature,
                "high_income_probability_percent": (
                    "고소득 예측 확률 (%)"
                ),
            },
        )

    figure.update_yaxes(
        range=[
            0,
            100,
        ]
    )

    figure.update_traces(
        hovertemplate=(
            f"{feature}: %{{x}}"
            "<br>고소득 예측 확률: %{y:.2f}%"
            "<extra></extra>"
        )
    )

    return figure

def plot_global_feature_importance(
    feature_importance: pd.DataFrame,
    *,
    top_n: int = 10,
) -> go.Figure:
    """전체 테스트셋 기준 permutation importance를 시각화한다."""

    _require_columns(
        feature_importance,
        [
            "feature",
            "importance_mean",
            "importance_std",
        ],
        "전체 모델 Feature Importance",
    )

    if feature_importance.empty:
        raise VisualizationError(
            "Feature Importance 결과가 비어 있습니다."
        )

    if top_n < 1:
        raise VisualizationError(
            "top_n은 1 이상이어야 합니다."
        )

    chart_data = (
        feature_importance
        .sort_values(
            "importance_mean",
            ascending=False,
        )
        .head(top_n)
        .sort_values(
            "importance_mean",
            ascending=True,
        )
        .copy()
    )

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=chart_data[
                "importance_mean"
            ],
            y=chart_data[
                "feature"
            ],
            orientation="h",
            error_x={
                "type": "data",
                "array": chart_data[
                    "importance_std"
                ],
                "visible": True,
            },
            hovertemplate=(
                "%{y}"
                "<br>Permutation Importance: %{x:.4f}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title="전체 모델 기준 예측 변수 중요도",
        xaxis_title=(
            "ROC-AUC Permutation Importance"
        ),
        yaxis_title="",
        showlegend=False,
    )

    return figure
