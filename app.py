"""Adult Census Income 연관성 분석·예측 웹 애플리케이션.

사용자 기능:
    1. 관심 변수와 통제 변수를 선택한 고소득 연관성 분석
    2. 사용자 입력 조건에 대한 고소득 확률 예측
    3. 개인 예측 설명
    4. What-if 시뮬레이션
    5. 전체 모델 기준 Feature Importance 확인

분석과 예측 계산은 src 모듈이 담당하며,
이 파일은 Streamlit UI와 결과 표시만 담당한다.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.association import (
    AnalysisRequest,
    AssociationError,
    analyze_association,
)
from src.config import (
    ANALYSIS_VARIABLE_TYPES,
    ANALYSIS_VARIABLES,
)
from src.data import load_and_clean
from src.modeling import (
    ModelingError,
    get_global_feature_importance,
    get_prediction_input_schema,
    predict_income_input,
    simulate_income_what_if,
)
from src.visualization import (
    VisualizationError,
    create_association_visualizations,
    plot_global_feature_importance,
    plot_prediction_explanation,
    plot_prediction_probability,
    plot_what_if_simulation,
)


# ============================================================
# 페이지 설정
# ============================================================

st.set_page_config(
    page_title="Adult Income Explorer",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 간단한 UI 스타일
# ============================================================

st.markdown(
    """
    <style>
    /* -------------------------------------------------------
       Global
    ------------------------------------------------------- */

    .stApp {
        background-color: #F7F6F3;
        color: #2C2C2C;
    }

    .block-container {
        max-width: 1240px;
        padding-top: 2.5rem;
        padding-bottom: 5rem;
    }

    h1, h2, h3 {
        color: #2C2C2C;
        letter-spacing: -0.025em;
    }

    h1 {
        font-size: 2.6rem !important;
        font-weight: 600 !important;
    }

    h2 {
        margin-top: 1.8rem !important;
    }

    /* -------------------------------------------------------
       Hero
    ------------------------------------------------------- */

    .hero-eyebrow {
        color: #7C8B6F;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        margin-bottom: 0.7rem;
    }

    .hero-title {
        color: #2C2C2C;
        font-size: 3rem;
        line-height: 1.15;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    .hero-title,
    h1,
    h2 {
        font-family:
            "Cormorant Garamond",
            "Noto Serif KR",
            Georgia,
            serif;
    }

    body,
    p,
    label,
    button,
    input,
    textarea,
    [data-testid="stMetric"] {
        font-family:
            -apple-system,
            BlinkMacSystemFont,
            "Pretendard",
            "Noto Sans KR",
            sans-serif;
    }

    .hero-description {
        color: #66635E;
        font-size: 1.05rem;
        line-height: 1.8;
        max-width: 850px;
        margin-bottom: 1.6rem;
    }

    /* -------------------------------------------------------
       Section labels
    ------------------------------------------------------- */

    .section-number {
        color: #7C8B6F;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        margin-bottom: 0.2rem;
    }

    .section-description {
        color: #77736D;
        font-size: 0.92rem;
        margin-top: -0.4rem;
        margin-bottom: 1rem;
    }

    /* -------------------------------------------------------
       Notes
    ------------------------------------------------------- */

    .interpretation-box {
        background-color: #EEEDE8;
        border-left: 3px solid #7C8B6F;
        padding: 1rem 1.2rem;
        border-radius: 4px;
        line-height: 1.65;
        margin-top: 1rem;
    }

    .question-box {
        background-color: #EEEDE8;
        padding: 1.4rem;
        border-radius: 8px;
        min-height: 160px;
    }

    .question-label {
        color: #7C8B6F;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }

    .question-text {
        color: #2C2C2C;
        font-size: 1.1rem;
        line-height: 1.7;
    }

    /* -------------------------------------------------------
       Streamlit components
    ------------------------------------------------------- */

    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.35);
        border-radius: 10px;
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.45);
    }

    div[data-baseweb="select"] > div {
        background-color: #FCFBF8;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.8rem;
    }

    hr {
        margin-top: 1.2rem !important;
        margin-bottom: 1.6rem !important;
    }

    hr {
        border-color: #DDDAD3 !important;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 사용자 표시용 변수 이름
# ============================================================

VARIABLE_LABELS = {
    "age": "나이",
    "workclass": "근무 형태",
    "education": "교육 수준",
    "marital-status": "혼인 상태",
    "occupation": "직업",
    "relationship": "가구 관계",
    "race": "인종",
    "sex": "성별",
    "capital-gain": "자본 이득",
    "capital-loss": "자본 손실",
    "hours-per-week": "주당 근무시간",
    "native-country": "출신 국가",
}


VARIABLE_TYPE_LABELS = {
    "binary": "이진형",
    "continuous": "연속형",
    "categorical": "범주형",
}


def _variable_label(
    variable: str,
) -> str:
    """변수명을 사용자 표시용 이름으로 변환한다."""

    korean = VARIABLE_LABELS.get(
        variable,
        variable,
    )

    return (
        f"{korean} ({variable})"
    )


# ============================================================
# 데이터 및 모델 메타데이터 캐시
# ============================================================

@st.cache_data
def load_service_data() -> pd.DataFrame:
    """서비스에서 공통으로 사용할 정제 Adult 데이터를 로딩한다."""

    return load_and_clean(
        save_output=False,
    )


@st.cache_data
def load_prediction_schema() -> dict:
    """예측 입력 폼 생성에 필요한 모델 스키마를 반환한다."""

    return (
        get_prediction_input_schema()
    )


@st.cache_data
def load_global_importance() -> pd.DataFrame:
    """현재 저장된 모델의 전체 permutation importance를 반환한다."""

    return (
        get_global_feature_importance()
    )


# ============================================================
# 공통 표시 함수
# ============================================================

def _format_p_value(
    value: float | None,
) -> str:
    """p-value를 화면에 적절한 문자열로 표시한다."""

    if value is None:
        return "추정 불가"

    value = float(
        value
    )

    if value < 0.001:
        return f"{value:.2e}"

    return f"{value:.4f}"


def _format_percent(
    value: float | None,
) -> str:
    """0~1 비율을 백분율 문자열로 변환한다."""

    if value is None:
        return "-"

    return (
        f"{float(value) * 100:.2f}%"
    )


def _display_interpretation_note(
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


# ============================================================
# 조정 전 결과
# ============================================================

def display_unadjusted_result(
    result: dict,
) -> None:
    """관심 변수 유형에 따라 조정 전 통계 결과를 표시한다."""

    analysis = result[
        "analysis"
    ]

    exposure_type = analysis[
        "exposure_type"
    ]

    unadjusted = analysis[
        "unadjusted"
    ]

    if exposure_type == "binary":
        metadata = unadjusted[
            "exposure_metadata"
        ]

        reference = metadata[
            "reference_level"
        ]

        comparison = metadata[
            "comparison_level"
        ]

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:
            st.metric(
                f"{reference} 고소득률",
                _format_percent(
                    unadjusted[
                        "reference_rate"
                    ]
                ),
            )

        with col2:
            st.metric(
                f"{comparison} 고소득률",
                _format_percent(
                    unadjusted[
                        "comparison_rate"
                    ]
                ),
            )

        with col3:
            st.metric(
                "고소득률 차이",
                (
                    f"{unadjusted['rate_difference'] * 100:+.2f}%p"
                ),
            )

        table = pd.DataFrame(
            [
                {
                    "통계량": "Risk Ratio",
                    "값": (
                        unadjusted[
                            "risk_ratio"
                        ]
                    ),
                },
                {
                    "통계량": "Odds Ratio",
                    "값": (
                        unadjusted[
                            "odds_ratio"
                        ]
                    ),
                },
                {
                    "통계량": "OR 95% CI 하한",
                    "값": (
                        unadjusted[
                            "odds_ratio_ci_95_low"
                        ]
                    ),
                },
                {
                    "통계량": "OR 95% CI 상한",
                    "값": (
                        unadjusted[
                            "odds_ratio_ci_95_high"
                        ]
                    ),
                },
                {
                    "통계량": "Cohen's h",
                    "값": (
                        unadjusted[
                            "cohens_h"
                        ]
                    ),
                },
                {
                    "통계량": "Fisher exact p-value",
                    "값": (
                        unadjusted[
                            "fisher_exact_p_value"
                        ]
                    ),
                },
            ]
        )

        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
        )

    elif exposure_type == "continuous":
        col1, col2 = (
            st.columns(2)
        )

        with col1:
            st.metric(
                "Point-biserial correlation",
                (
                    f"{unadjusted['correlation']:.4f}"
                ),
            )

        with col2:
            st.metric(
                "p-value",
                _format_p_value(
                    unadjusted[
                        "p_value"
                    ]
                ),
            )

        st.caption(
            "고소득 여부(0/1)와 연속형 관심 변수 사이의 "
            "조정 전 연관성을 나타냅니다."
        )

    elif exposure_type == "categorical":
        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:
            st.metric(
                "Chi-square",
                (
                    f"{unadjusted['chi2_statistic']:.3f}"
                ),
            )

        with col2:
            st.metric(
                "p-value",
                _format_p_value(
                    unadjusted[
                        "chi2_p_value"
                    ]
                ),
            )

        with col3:
            st.metric(
                "자유도",
                unadjusted[
                    "degrees_of_freedom"
                ],
            )

        group_table = pd.DataFrame(
            unadjusted[
                "groups"
            ]
        )

        group_table[
            "고소득률 (%)"
        ] = (
            group_table[
                "target_rate"
            ]
            * 100
        )

        group_table = (
            group_table
            .drop(
                columns=[
                    "target_rate",
                ]
            )
            .rename(
                columns={
                    "n": "표본 수",
                }
            )
        )

        st.dataframe(
            group_table,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# Logistic Regression 결과
# ============================================================

def display_adjusted_result(
    result: dict,
) -> None:
    """조정된 관심 변수 Odds Ratio 결과를 표시한다."""

    adjusted = (
        result[
            "analysis"
        ][
            "adjusted"
        ]
    )

    effects = (
        adjusted[
            "exposure_effects"
        ]
    )

    rows: list[dict] = []

    for effect in effects:
        estimable = effect.get(
            "estimable",
            True,
        )

        rows.append(
            {
                "항목": effect[
                    "term"
                ],
                "Adjusted OR": (
                    effect[
                        "odds_ratio"
                    ]
                    if estimable
                    else None
                ),
                "95% CI 하한": (
                    effect[
                        "ci_95_low"
                    ]
                    if estimable
                    else None
                ),
                "95% CI 상한": (
                    effect[
                        "ci_95_high"
                    ]
                    if estimable
                    else None
                ),
                "p-value": (
                    effect[
                        "p_value"
                    ]
                ),
                "추정 가능": (
                    "예"
                    if estimable
                    else "아니오"
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(
            rows
        ),
        width="stretch",
        hide_index=True,
    )

    warnings = adjusted.get(
        "estimation_warnings",
        [],
    )

    for warning in warnings:
        st.warning(
            warning
        )

    diagnostics = adjusted[
        "model_diagnostics"
    ]

    with st.expander(
        "Logistic Regression 모델 진단"
    ):
        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:
            st.metric(
                "분석 표본",
                diagnostics[
                    "n_observations"
                ],
            )

        with col2:
            st.metric(
                "Pseudo R²",
                (
                    f"{diagnostics['pseudo_r_squared']:.4f}"
                ),
            )

        with col3:
            st.metric(
                "AIC",
                (
                    f"{diagnostics['aic']:.2f}"
                ),
            )


# ============================================================
# PSM 결과
# ============================================================

def display_psm_result(
    psm: dict,
) -> None:
    """PSM 결과와 균형 진단을 표시한다."""

    result = psm[
        "result"
    ]

    matching = result[
        "matching"
    ]

    outcome = result[
        "outcome_comparison"
    ]

    balance = result[
        "balance"
    ]

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:
        st.metric(
            "매칭 쌍",
            matching[
                "matched_pairs"
            ],
        )

    with col2:
        st.metric(
            "매칭 후 고소득률 차이",
            (
                f"{outcome['matched_rate_difference'] * 100:+.2f}%p"
            ),
        )

    with col3:
        st.metric(
            "McNemar p-value",
            _format_p_value(
                outcome[
                    "mcnemar_p_value"
                ]
            ),
        )

    col1, col2 = (
        st.columns(2)
    )

    with col1:
        st.metric(
            "매칭 전 최대 SMD",
            (
                "-"
                if balance[
                    "max_smd_before"
                ]
                is None
                else (
                    f"{balance['max_smd_before']:.3f}"
                )
            ),
        )

    with col2:
        st.metric(
            "매칭 후 최대 SMD",
            (
                "-"
                if balance[
                    "max_smd_after"
                ]
                is None
                else (
                    f"{balance['max_smd_after']:.3f}"
                )
            ),
        )

    if balance[
        "balanced_under_threshold"
    ]:
        st.success(
            "매칭 후 모든 평가 가능한 공변량의 "
            "절대 SMD가 기준값 미만입니다."
        )
    else:
        st.warning(
            "매칭 후에도 일부 공변량의 SMD가 "
            "균형 기준을 충족하지 못했습니다."
        )

    with st.expander(
        "PSM 세부 정보"
    ):
        st.write(
            {
                "comparison retention rate": (
                    matching[
                        "comparison_retention_rate"
                    ]
                ),
                "caliper": (
                    matching[
                        "caliper"
                    ]
                ),
                "common support": (
                    matching[
                        "common_support_lower"
                    ],
                    matching[
                        "common_support_upper"
                    ],
                ),
                "mean match distance": (
                    matching[
                        "mean_match_distance"
                    ]
                ),
            }
        )


# ============================================================
# 연관성 분석 페이지
# ============================================================

def association_page(
    df: pd.DataFrame,
) -> None:
    """사용자 선택형 고소득 연관성 분석 UI."""

    st.header(
        "연관성 분석"
    )

    st.markdown(
        """
        <div class="section-description">
            하나의 관심 변수를 선택하고, 필요한 경우 통제 변수를 추가하여
            고소득 여부와의 조정 전·후 연관성을 비교합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


    settings_col, question_col = st.columns(
        [0.9, 1.4],
        gap="large",
    )


    with settings_col:
        with st.container(
            border=True
        ):
            st.subheader(
                "분석 설정"
            )

            exposure = st.selectbox(
                "관심 변수",
                options=list(
                    ANALYSIS_VARIABLES
                ),
                format_func=_variable_label,
                key="association_exposure",
            )

            exposure_type = (
                ANALYSIS_VARIABLE_TYPES[
                    exposure
                ]
            )

            st.caption(
                "변수 유형 · "
                f"{VARIABLE_TYPE_LABELS[exposure_type]}"
            )

            control_options = [
                variable
                for variable
                in ANALYSIS_VARIABLES
                if variable != exposure
            ]

            controls = st.multiselect(
                "통제 변수",
                options=control_options,
                format_func=_variable_label,
                key="association_controls",
                placeholder=(
                    "조정할 변수를 선택하세요"
                ),
            )

            psm_available = (
                exposure_type == "binary"
                and bool(
                    controls
                )
            )

            if psm_available:
                include_psm = st.toggle(
                    "PSM 추가 분석",
                    value=False,
                    help=(
                        "선택한 통제 변수의 분포를 "
                        "성향점수매칭으로 추가 조정합니다."
                    ),
                )

            else:
                include_psm = False

            analyze_button = st.button(
                "분석 실행",
                type="primary",
                width="stretch",
            )

    with question_col:
        exposure_label = VARIABLE_LABELS.get(
            exposure,
            exposure,
        )

        control_labels = [
            VARIABLE_LABELS.get(
                control,
                control,
            )
            for control in controls
        ]

        if control_labels:
            controls_text = ", ".join(
                control_labels
            )

            question = (
                f"선택한 통제 변수({controls_text})를 고려한 뒤에도 "
                f"{exposure_label}과 고소득 여부 사이의 "
                "연관성이 나타나는가?"
            )

        else:
            question = (
                f"{exposure_label}과 고소득 여부 사이에는 "
                "어떤 연관성이 있는가?"
            )

        with st.container(
            border=True
        ):
            st.markdown(
                """
                <div class="section-number">
                    CURRENT QUESTION
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"### {question}"
            )

            st.write("")

            if exposure_type == "continuous":
                st.caption(
                    "조정 전에는 연속형 변수와 고소득 여부의 "
                    "관계를 확인하고, 조정 후에는 Logistic Regression의 "
                    "Odds Ratio를 확인합니다."
                )

            elif exposure_type == "binary":
                st.caption(
                    "두 집단의 고소득률을 비교한 뒤 "
                    "Logistic Regression으로 통제 변수를 조정합니다."
                )

            else:
                st.caption(
                    "범주별 고소득률을 비교한 뒤 "
                    "각 범주와 기준 범주의 Adjusted Odds Ratio를 확인합니다."
                )
    if analyze_button:
        request = AnalysisRequest(
            exposure=exposure,
            controls=tuple(
                controls
            ),
            include_psm=(
                include_psm
            ),
        )

        try:
            with st.spinner(
                "연관성 분석을 수행하고 있습니다..."
            ):
                result = (
                    analyze_association(
                        df,
                        request,
                    )
                )

                figures = (
                    create_association_visualizations(
                        result
                    )
                )

        except (
            AssociationError,
            VisualizationError,
            ValueError,
        ) as exc:
            st.error(
                str(exc)
            )
            return

        st.session_state[
            "association_result"
        ] = result

        st.session_state[
            "association_figures"
        ] = figures

    result = st.session_state.get(
        "association_result"
    )

    figures = st.session_state.get(
        "association_figures"
    )

    if (
        result is None
        or figures is None
    ):
        return

    request_result = result[
        "request"
    ]

    analysis = result[
        "analysis"
    ]

    st.markdown("#### 분석 방법")

    if exposure_type == "continuous":
        st.caption(
            "조정 전: Point-biserial correlation · "
            "조정 후: Logistic Regression"
        )

    elif exposure_type == "binary":
        st.caption(
            "조정 전: 집단별 고소득률 비교 · "
            "조정 후: Logistic Regression"
        )

    else:
        st.caption(
            "조정 전: 범주별 고소득률 + Chi-square · "
            "조정 후: Adjusted Odds Ratio"
        )

    st.divider()

    st.markdown(
        '<div class="section-number">01 · UNADJUSTED</div>',
        unsafe_allow_html=True,
    )

    st.subheader(
        "조정 전 연관성"
    )
    
    st.markdown(
        '<div class="section-number">02 · ADJUSTED</div>',
        unsafe_allow_html=True,
    )

    st.subheader(
        "통제 변수 조정 후"
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:
        st.metric(
            "분석 표본",
            f"{analysis['sample_size']:,}",
            border=True,
        )

    with col2:
        st.metric(
            "제외된 행",
            f"{analysis['rows_excluded_due_to_missing']:,}",
            border=True,
        )

    with col3:
        st.metric(
            "통제 변수",
            f"{len(request_result['controls'])}개",
            border=True,
        )

    selected_controls = (
        ", ".join(
            _variable_label(
                control
            )
            for control
            in request_result[
                "controls"
            ]
        )
        if request_result[
            "controls"
        ]
        else "없음"
    )

    st.caption(
        "관심 변수: "
        f"{_variable_label(request_result['exposure'])} · "
        "통제 변수: "
        f"{selected_controls}"
    )

    # --------------------------------------------------------
    # 조정 전
    # --------------------------------------------------------

    st.subheader(
        "1. 조정 전 연관성"
    )

    display_unadjusted_result(
        result
    )

    st.plotly_chart(
        figures[
            "unadjusted"
        ],
        width="stretch",
    )

    # --------------------------------------------------------
    # 조정 후
    # --------------------------------------------------------

    st.subheader(
        "2. 통제 변수 조정 후 연관성"
    )

    display_adjusted_result(
        result
    )

    st.plotly_chart(
        figures[
            "adjusted"
        ],
        width="stretch",
    )

    if not request_result[
        "controls"
    ]:
        st.info(
            "통제 변수를 선택하지 않았으므로 "
            "이 Logistic Regression 결과에는 추가 조정 변수가 없습니다."
        )

    # --------------------------------------------------------
    # PSM
    # --------------------------------------------------------

    psm = analysis.get(
        "psm"
    )

    if psm is not None:
        st.subheader(
            "3. 성향점수매칭(PSM)"
        )

        display_psm_result(
            psm
        )

        if (
            "psm_balance"
            in figures
        ):
            st.plotly_chart(
                figures[
                    "psm_balance"
                ],
                width="stretch",
            )

        _display_interpretation_note(
            psm[
                "result"
            ][
                "interpretation_note"
            ]
        )

    _display_interpretation_note(
        result[
            "interpretation_note"
        ]
    )


# ============================================================
# 예측 입력 Widget
# ============================================================

def _continuous_input_widget(
    feature: str,
    info: dict,
) -> float:
    """연속형 모델 피처의 Streamlit 입력 Widget을 생성한다."""

    minimum = float(
        info[
            "minimum"
        ]
    )

    maximum = float(
        info[
            "maximum"
        ]
    )

    reference = float(
        info[
            "reference_value"
        ]
    )

    integer_like = all(
        value.is_integer()
        for value in [
            minimum,
            maximum,
            reference,
        ]
    )

    if integer_like:
        return float(
            st.number_input(
                _variable_label(
                    feature
                ),
                min_value=int(
                    minimum
                ),
                max_value=int(
                    maximum
                ),
                value=int(
                    reference
                ),
                step=1,
                key=(
                    f"prediction_{feature}"
                ),
            )
        )

    return float(
        st.number_input(
            _variable_label(
                feature
            ),
            min_value=minimum,
            max_value=maximum,
            value=reference,
            key=(
                f"prediction_{feature}"
            ),
        )
    )


def _categorical_input_widget(
    feature: str,
    info: dict,
):
    """범주형 모델 피처의 Streamlit 선택 Widget을 생성한다."""

    levels = list(
        info[
            "levels"
        ]
    )

    reference = info[
        "reference_value"
    ]

    try:
        default_index = (
            levels.index(
                reference
            )
        )
    except ValueError:
        default_index = 0

    return st.selectbox(
        _variable_label(
            feature
        ),
        options=levels,
        index=default_index,
        key=(
            f"prediction_{feature}"
        ),
    )


# ============================================================
# 예측 페이지
# ============================================================

def prediction_page() -> None:
    """개별 입력의 고소득 예측과 모델 설명 UI."""

    st.header(
        "고소득 예측"
    )

    st.write(
        "개인의 조건을 입력하면 학습된 머신러닝 모델이 "
        "연 소득 5만 달러 초과 확률을 예측합니다."
    )

    st.caption(
        "이 기능은 통계적 연관성 분석과 별개입니다. "
        "입력값은 실제 예측 조건으로 사용됩니다."
    )

    try:
        schema = (
            load_prediction_schema()
        )
    except ModelingError as exc:
        st.error(
            str(exc)
        )
        return

    user_input: dict = {}

    feature_columns = (
        schema[
            "feature_columns"
        ]
    )


    def render_prediction_feature(
        feature: str,
    ) -> None:
        """모델 스키마에 따라 하나의 예측 입력 Widget을 생성한다."""

        info = (
            schema[
                "features"
            ][feature]
        )

        if (
            info[
                "type"
            ]
            == "continuous"
        ):
            user_input[
                feature
            ] = (
                _continuous_input_widget(
                    feature,
                    info,
                )
            )

        else:
            user_input[
                feature
            ] = (
                _categorical_input_widget(
                    feature,
                    info,
                )
            )


    with st.form(
        "prediction_form",
        border=False,
    ):

        # ========================================================
        # 01. 기본 정보
        # ========================================================

        with st.container(
            border=True
        ):
            st.markdown(
                """
                <div class="section-number">
                    01 · PROFILE
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader(
                "기본 정보"
            )

            left, right = (
                st.columns(2)
            )

            with left:
                render_prediction_feature(
                    "age"
                )

                render_prediction_feature(
                    "race"
                )

                render_prediction_feature(
                    "relationship"
                )

            with right:
                render_prediction_feature(
                    "sex"
                )

                render_prediction_feature(
                    "marital-status"
                )

                render_prediction_feature(
                    "native-country"
                )


        # ========================================================
        # 02. 교육 및 직업
        # ========================================================

        with st.container(
            border=True
        ):
            st.markdown(
                """
                <div class="section-number">
                    02 · EDUCATION & WORK
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader(
                "교육 및 직업"
            )

            left, right = (
                st.columns(2)
            )

            with left:
                render_prediction_feature(
                    "education"
                )

                render_prediction_feature(
                    "occupation"
                )

            with right:
                render_prediction_feature(
                    "workclass"
                )

                render_prediction_feature(
                    "hours-per-week"
                )


        # ========================================================
        # 03. 자본 정보
        # ========================================================

        with st.container(
            border=True
        ):
            st.markdown(
                """
                <div class="section-number">
                    03 · CAPITAL
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader(
                "자본 정보"
            )

            left, right = (
                st.columns(2)
            )

            with left:
                render_prediction_feature(
                    "capital-gain"
                )

            with right:
                render_prediction_feature(
                    "capital-loss"
                )
                
        st.write("")
        
        button_left, button_center, button_right = (
            st.columns(
                [
                    1,
                    2,
                    1,
                ]
            )
        )

        with button_center:
            predict_button = (
                st.form_submit_button(
                    "예측 실행",
                    type="primary",
                    width="stretch",
                )
            )
            
    if predict_button:
        try:
            with st.spinner(
                "고소득 확률을 예측하고 있습니다..."
            ):
                prediction_result = (
                    predict_income_input(
                        user_input
                    )
                )

        except ModelingError as exc:
            st.error(
                str(exc)
            )
            return

        st.session_state[
            "prediction_result"
        ] = prediction_result

        st.session_state[
            "prediction_input"
        ] = dict(
            user_input
        )
    prediction_result = (
        st.session_state.get(
            "prediction_result"
        )
    )

    prediction_input = (
        st.session_state.get(
            "prediction_input"
        )
    )

    if (
        prediction_result is None
        or prediction_input is None
    ):
        return

    st.divider()

    prediction = (
        prediction_result[
            "prediction"
        ]
    )

    # --------------------------------------------------------
    # 예측 결과
    # --------------------------------------------------------

    st.subheader(
        "예측 결과"
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:
        st.metric(
            "예측 클래스",
            prediction[
                "prediction_label"
            ],
        )

    with col2:
        st.metric(
            "고소득 예측 확률",
            (
                f"{prediction['high_income_probability'] * 100:.2f}%"
            ),
        )

    try:
        probability_figure = (
            plot_prediction_probability(
                prediction_result
            )
        )

        st.plotly_chart(
            probability_figure,
            width="stretch",
        )

    except VisualizationError as exc:
        st.warning(
            f"예측 확률 그래프를 표시하지 못했습니다: {exc}"
        )

    # --------------------------------------------------------
    # 개인별 설명
    # --------------------------------------------------------

    st.subheader(
        "개인 입력 기준 예측 설명"
    )

    st.caption(
        "각 변수의 현재 값을 학습 데이터의 대표값으로 "
        "하나씩 변경했을 때 예측 확률이 얼마나 달라지는지 비교합니다."
    )

    try:
        explanation_figure = (
            plot_prediction_explanation(
                prediction_result
            )
        )

        st.plotly_chart(
            explanation_figure,
            width="stretch",
        )

    except VisualizationError as exc:
        st.warning(
            f"개인 예측 설명 그래프를 표시하지 못했습니다: {exc}"
        )

    _display_interpretation_note(
        prediction_result[
            "explanation"
        ][
            "interpretation_note"
        ]
    )

    # --------------------------------------------------------
    # 전체 모델 Feature Importance
    # --------------------------------------------------------

    st.subheader(
        "전체 모델 기준 예측 변수 중요도"
    )

    st.caption(
        "개별 사용자가 아니라 전체 테스트 데이터에서 "
        "각 변수를 섞었을 때 ROC-AUC가 얼마나 감소하는지 측정한 "
        "Permutation Importance입니다."
    )

    try:
        importance = (
            load_global_importance()
        )

        importance_figure = (
            plot_global_feature_importance(
                importance
            )
        )

        st.plotly_chart(
            importance_figure,
            width="stretch",
        )

    except (
        ModelingError,
        VisualizationError,
    ) as exc:
        st.warning(
            f"전체 모델 중요도를 표시하지 못했습니다: {exc}"
        )

    # --------------------------------------------------------
    # What-if
    # --------------------------------------------------------

    st.subheader(
        "What-if Simulation"
    )

    st.write(
        "다른 입력값은 그대로 유지하고 "
        "한 변수만 변경하여 모델 예측 확률의 변화를 확인합니다."
    )

    what_if_feature = st.selectbox(
        "변경할 변수",
        options=feature_columns,
        format_func=_variable_label,
        key="what_if_feature",
    )

    if st.button(
        "What-if 실행",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "시나리오별 예측을 계산하고 있습니다..."
            ):
                what_if = (
                    simulate_income_what_if(
                        prediction_input,
                        feature=(
                            what_if_feature
                        ),
                    )
                )

                what_if_figure = (
                    plot_what_if_simulation(
                        what_if
                    )
                )

        except (
            ModelingError,
            VisualizationError,
        ) as exc:
            st.error(
                str(exc)
            )

        else:
            st.session_state[
                "what_if_result"
            ] = what_if

            st.session_state[
                "what_if_figure"
            ] = what_if_figure

            st.session_state[
                "what_if_result_feature"
            ] = what_if_feature

    what_if_figure = (
        st.session_state.get(
            "what_if_figure"
        )
    )

    what_if_result_feature = (
        st.session_state.get(
            "what_if_result_feature"
        )
    )

    if (
        what_if_figure is not None
        and what_if_result_feature
        == what_if_feature
    ):
        st.plotly_chart(
            what_if_figure,
            width="stretch",
        )

        st.caption(
            "What-if 결과는 한 변수의 입력값을 바꾸었을 때 "
            "현재 모델의 예측이 어떻게 달라지는지를 보여주며, "
            "해당 변수를 실제로 변화시켰을 때 발생하는 "
            "인과효과를 의미하지 않습니다."
        )

    _display_interpretation_note(
        prediction_result[
            "interpretation_note"
        ]
    )


# ============================================================
# 앱 실행
# ============================================================

def render_hero() -> None:
    """서비스의 상단 소개 영역을 표시한다."""

    st.markdown(
        """
        <div class="hero-eyebrow">
            SKALA · ADULT CENSUS INCOME
        </div>

        <div class="hero-title">
            고소득 연관성 분석 & 예측
        </div>

        <div class="hero-description">
            관심 변수와 통제 변수를 직접 선택해 고소득 여부와의 연관성을 탐색하고,
            개인의 조건에 따른 머신러닝 예측 확률과 예측 근거를 확인합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

def main() -> None:
    """Streamlit 웹 애플리케이션을 실행한다."""

    render_hero()

    try:
        df = (
            load_service_data()
        )
    except Exception as exc:
        st.error(
            "Adult 데이터를 불러오지 못했습니다: "
            f"{exc}"
        )
        st.stop()

    nav_left, nav_center, nav_right = st.columns(
        [1, 3, 1]
    )

    with nav_center:
        mode = st.segmented_control(
            "서비스 선택",
            options=[
                "연관성 분석",
                "고소득 예측",
            ],
            default="연관성 분석",
            label_visibility="collapsed",
            width="stretch",
        )

    st.divider()

    if mode == "연관성 분석":
        association_page(
            df
        )

    else:
        prediction_page()

if __name__ == "__main__":
    main()