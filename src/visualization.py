"""Adult Income 웹 서비스의 사용자용 시각화.

이 모듈은 분석 결과를 다시 계산하거나 파일에서 읽지 않는다.

association.py와 modeling.py가 반환한 결과 객체와
필요한 DataFrame을 받아 Plotly Figure로 변환한다.

주요 역할:
    1. Logistic Regression 기반 예상 고소득 비율 시각화
    2. PSM 수행 시 매칭 전후 공변량 균형 시각화
    3. 개인 예측 설명·What-if·전체 중요도 시각화

모든 함수는 파일을 저장하지 않고 Figure 객체를 반환한다.
웹 UI는 반환된 Figure를 직접 표시한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.labels import (
    CATEGORY_VALUE_LABELS,
    VARIABLE_LABELS,
)


class VisualizationError(ValueError):
    """시각화 입력이나 결과 구조가 올바르지 않을 때 발생하는 오류."""


# ============================================================
# 사용자 표시용 라벨 함수
# ============================================================

def _variable_label(
    variable: str,
) -> str:
    return VARIABLE_LABELS.get(
        variable,
        variable,
    )


def _category_label(
    variable: str,
    value,
) -> str:
    """그래프에서는 한글만 표시한다."""

    return (
        CATEGORY_VALUE_LABELS
        .get(variable, {})
        .get(value, str(value))
    )


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
# 예상 고소득 비율 시각화
# ============================================================

def plot_adjusted_probability(
    result: dict,
) -> go.Figure:
    """관심 변수 값별 예상 고소득 비율을 표시한다."""

    _validate_association_result(
        result
    )

    request = result["request"]
    analysis = result["analysis"]

    exposure = request["exposure"]
    exposure_type = analysis[
        "exposure_type"
    ]

    exposure_label = (
        _variable_label(
            exposure
        )
    )

    adjusted = analysis[
        "adjusted"
    ]

    probability_result = (
        adjusted.get(
            "adjusted_probabilities"
        )
    )

    if not isinstance(
        probability_result,
        dict,
    ):
        raise VisualizationError(
            "예상 비율 결과가 없습니다."
        )

    records = probability_result.get(
        "records",
        [],
    )

    if not records:
        raise VisualizationError(
            "시각화할 예상 비율이 없습니다."
        )

    chart_data = pd.DataFrame(
        records
    )

    chart_data[
        "probability_percent"
    ] = (
        chart_data[
            "adjusted_probability"
        ]
        * 100
    )

    # ========================================================
    # 연속형
    # ========================================================

    if exposure_type == "continuous":

        chart_data[
            "exposure_value"
        ] = pd.to_numeric(
            chart_data[
                "exposure_value"
            ],
            errors="raise",
        )

        chart_data = (
            chart_data
            .sort_values(
                "exposure_value"
            )
        )

        figure = px.line(
            chart_data,
            x="exposure_value",
            y="probability_percent",
            markers=True,
            labels={
                "exposure_value": (
                    exposure_label
                ),
                "probability_percent": (
                    "연 소득 5만 달러 초과 예상 비율 (%)"
                ),
            },
        )

        figure.update_traces(
            line={
                "color": "#7C8B6F",
                "width": 3,
            },
            marker={
                "color": "#7C8B6F",
                "size": 8,
            },
            hovertemplate=(
                f"{exposure_label}: %{{x}}"
                "<br>예상 비율: %{y:.1f}%"
                "<extra></extra>"
            ),
        )

    # ========================================================
    # 이진형 / 범주형
    # ========================================================

    else:

        chart_data[
            "display_value"
        ] = chart_data[
            "exposure_value"
        ].map(
            lambda value: (
                _category_label(
                    exposure,
                    value,
                )
            )
        )

        figure = px.bar(
            chart_data,
            x="display_value",
            y="probability_percent",
            labels={
                "display_value": (
                    exposure_label
                ),
                "probability_percent": (
                    "연 소득 5만 달러 초과 예상 비율 (%)"
                ),
            },
        )

        figure.update_traces(
            marker_color="#7C8B6F",
            hovertemplate=(
                f"{exposure_label}: %{{x}}"
                "<br>예상 비율: %{y:.1f}%"
                "<extra></extra>"
            ),
        )

    figure.update_yaxes(
        range=[0, 100],
        title=(
            "연 소득 5만 달러 초과 예상 비율 (%)"
        ),
    )

    figure.update_xaxes(
        title=exposure_label,
    )

    return _apply_user_chart_theme(
        figure,
        height=430,
    )


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
        xaxis_title=(
            "Absolute Standardized "
            "Mean Difference"
        ),
        yaxis_title="",
    )

    return figure


# ============================================================
# 연관성 분석 시각화 묶음
# ============================================================

def create_association_visualizations(
    result: dict,
) -> dict[str, go.Figure]:
    """현재 웹 UI에서 실제 사용하는 연관성 Figure만 생성한다."""

    figures = {
        "adjusted_probability": (
            plot_adjusted_probability(
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


# ============================================================
# 개인 예측 설명 시각화
# ============================================================

def plot_prediction_explanation(
    result: dict,
    *,
    top_n: int = 8,
) -> go.Figure:
    """현재 입력값을 대표값으로 바꿨을 때의 예측 확률 변화를 표시한다."""

    explanation = result.get(
        "explanation",
        {},
    )

    features = explanation.get(
        "features",
    )

    if not features:
        raise VisualizationError(
            "개인 예측 설명 데이터가 없습니다."
        )

    chart_data = pd.DataFrame(
        features
    ).copy()

    chart_data[
        "absolute_impact"
    ] = (
        chart_data[
            "impact_percentage_points"
        ].abs()
    )

    chart_data = (
        chart_data
        .sort_values(
            "absolute_impact",
            ascending=False,
        )
        .head(top_n)
        .sort_values(
            "impact_percentage_points",
            ascending=True,
        )
        .copy()
    )

    chart_data[
        "feature_label"
    ] = (
        chart_data[
            "feature"
        ]
        .map(
            VARIABLE_LABELS
        )
        .fillna(
            chart_data[
                "feature"
            ]
        )
    )

    def format_impact(
        value: float,
    ) -> str:
        """작은 영향값도 화면에서 구분할 수 있도록 표시한다."""

        if abs(value) < 0.001:
            return "0.000%p"

        if abs(value) < 0.01:
            return f"{value:+.3f}%p"

        return f"{value:+.2f}%p"

    chart_data[
        "impact_text"
    ] = chart_data[
        "impact_percentage_points"
    ].map(
        format_impact
    )

    max_abs_impact = float(
        chart_data[
            "absolute_impact"
        ].max()
    )

    axis_limit = max(
        max_abs_impact * 1.30,
        0.01,
    )

    figure = px.bar(
        chart_data,
        x="impact_percentage_points",
        y="feature_label",
        orientation="h",
        text="impact_text",
        custom_data=[
            "current_value",
            "reference_value",
        ],
        labels={
            "feature_label": "",
            "impact_percentage_points": (
                "대표적인 값과 비교한 예측 확률 차이 (%p)"
            ),
        },
    )

    figure.add_vline(
        x=0,
        line_dash="dash",
        line_color="#2C2C2C",
    )

    figure.update_traces(
        marker_color="#7C8B6F",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b>"
            "<br>예측 확률 차이: %{x:+.3f}%p"
            "<br>현재 입력값: %{customdata[0]}"
            "<br>대표적인 값: %{customdata[1]}"
            "<extra></extra>"
        ),
    )

    figure.update_xaxes(
        range=[
            -axis_limit,
            axis_limit,
        ],
        title=(
            "대표적인 값과 비교한 예측 확률 차이 (%p)"
        ),
        gridcolor="#E8E6E1",
        zeroline=False,
    )

    figure.update_yaxes(
        title="",
        gridcolor="rgba(0,0,0,0)",
    )

    figure.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin={
            "l": 20,
            "r": 70,
            "t": 10,
            "b": 55,
        },
    )

    return figure


# ============================================================
# What-if 시각화
# ============================================================

def plot_what_if_simulation(
    what_if: pd.DataFrame,
    *,
    feature_label: str | None = None,
    category_labels: dict | None = None,
    current_value=None,
) -> go.Figure:
    """한 항목만 변경했을 때 예측 확률 변화를 표시한다."""

    _require_columns(
        what_if,
        [
            "feature",
            "value",
            "high_income_probability_percent",
        ],
        "조건 변경 시뮬레이션",
    )

    if what_if.empty:
        raise VisualizationError(
            "조건 변경 결과가 비어 있습니다."
        )

    feature = str(
        what_if["feature"].iloc[0]
    )

    display_feature = (
        feature_label
        if feature_label is not None
        else feature
    )

    category_labels = (
        category_labels or {}
    )

    numeric_values = pd.to_numeric(
        what_if["value"],
        errors="coerce",
    )

    # ========================================================
    # 연속형
    # ========================================================

    if numeric_values.notna().all():

        chart_data = what_if.copy()

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
            y="high_income_probability_percent",
            markers=True,
            title=(
                f"{display_feature}를 바꿨을 때의 예측 변화"
            ),
            labels={
                "value": display_feature,
                "high_income_probability_percent": (
                    "연 소득 5만 달러 초과 예측 확률 (%)"
                ),
            },
        )

        if current_value is not None:
            try:
                current_numeric = float(
                    current_value
                )

                figure.add_vline(
                    x=current_numeric,
                    line_dash="dash",
                    annotation_text="현재 입력값",
                    annotation_position="top",
                )

            except (
                TypeError,
                ValueError,
            ):
                pass

    # ========================================================
    # 범주형
    # ========================================================

    else:

        chart_data = what_if.copy()

        def format_category(
            value,
        ) -> str:
            korean = (
                category_labels.get(
                    value
                )
            )

            if korean is None:
                label = str(
                    value
                )

            else:
                label = (
                    f"{korean} ({value})"
                )

            if (
                current_value is not None
                and value == current_value
            ):
                return (
                    f"{label} · 현재"
                )

            return label

        chart_data[
            "value_label"
        ] = chart_data[
            "value"
        ].map(
            format_category
        )

        figure = px.bar(
            chart_data,
            x="value_label",
            y="high_income_probability_percent",
            title=(
                f"{display_feature}을(를) 바꿨을 때의 예측 변화"
            ),
            labels={
                "value_label": display_feature,
                "high_income_probability_percent": (
                    "연 소득 5만 달러 초과 예측 확률 (%)"
                ),
            },
        )

    figure.update_yaxes(
        range=[0, 100],
        title=(
            "연 소득 5만 달러 초과 예측 확률 (%)"
        ),
    )

    figure.update_xaxes(
        title=display_feature,
    )

    figure.update_traces(
        hovertemplate=(
            f"<b>{display_feature}</b>: %{{x}}"
            "<br>예측 확률: %{y:.1f}%"
            "<extra></extra>"
        )
    )

    figure.update_layout(
        showlegend=False,
        height=460,
        margin={
            "l": 20,
            "r": 30,
            "t": 70,
            "b": 70,
        },
    )

    return figure


# ============================================================
# 전체 Feature Importance 시각화
# ============================================================

def plot_global_feature_importance(
    feature_importance: pd.DataFrame,
    *,
    top_n: int = 10,
) -> go.Figure:
    """전체 테스트 데이터 기준 예측 중요도를 시각화한다."""

    _require_columns(
        feature_importance,
        [
            "feature",
            "importance_mean",
            "importance_std",
        ],
        "전체 모델 예측 중요도",
    )

    if feature_importance.empty:
        raise VisualizationError(
            "예측 중요도 결과가 비어 있습니다."
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

    chart_data[
        "feature_label"
    ] = (
        chart_data[
            "feature"
        ]
        .map(
            VARIABLE_LABELS
        )
        .fillna(
            chart_data[
                "feature"
            ]
        )
    )

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=chart_data[
                "importance_mean"
            ],
            y=chart_data[
                "feature_label"
            ],
            marker={
                "color": "#7C8B6F",
            },
            orientation="h",
            error_x={
                "type": "data",
                "array": chart_data[
                    "importance_std"
                ],
                "visible": True,
            },
            hovertemplate=(
                "<b>%{y}</b>"
                "<br>예측 중요도: %{x:.4f}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        yaxis_title="",
        showlegend=False,
        height=500,
        margin={
            "l": 20,
            "r": 40,
            "t": 70,
            "b": 60,
        },
    )

    return _apply_user_chart_theme(
        figure,
        height=460,
    )


# ============================================================
# 공통 Plotly 테마
# ============================================================

def _apply_user_chart_theme(
    figure: go.Figure,
    *,
    height: int = 420,
) -> go.Figure:
    """웹 서비스용 공통 Plotly 스타일."""

    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "family": (
                "Pretendard, Noto Sans KR, "
                "Apple SD Gothic Neo, sans-serif"
            ),
            "color": "#4B4B4B",
            "size": 13,
        },
        height=height,
        margin={
            "l": 30,
            "r": 30,
            "t": 20,
            "b": 55,
        },
        showlegend=False,
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "font_color": "#2C2C2C",
        },
    )

    figure.update_xaxes(
        showline=False,
        zeroline=False,
        gridcolor="#E8E6E1",
        tickfont={
            "color": "#6B7280",
        },
        title_font={
            "color": "#6B7280",
        },
    )

    figure.update_yaxes(
        showline=False,
        zeroline=False,
        gridcolor="#E8E6E1",
        tickfont={
            "color": "#6B7280",
        },
        title_font={
            "color": "#6B7280",
        },
    )

    return figure