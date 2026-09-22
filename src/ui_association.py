"""연 소득 관계 분석 Streamlit UI.

통계 계산은 src.association과 src.statistics가 담당하고,
이 모듈은 관계 분석 입력·결과 표시·오류 안내만 담당한다.
"""

from __future__ import annotations

import logging

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
from src.labels import (
    CATEGORY_VALUE_LABELS,
    VARIABLE_LABELS,
)
from src.ui_common import (
    category_value_label,
    display_interpretation_note,
    display_plotly_chart,
    format_p_value,
    format_percent,
    result_value_label,
)
from src.visualization import (
    VisualizationError,
    create_association_visualizations,
)


logger = logging.getLogger(__name__)


def display_unadjusted_result(
    result: dict,
) -> None:
    """다른 조건을 고려하기 전 관찰된 관계를 사용자 친화적으로 표시한다."""

    request = result["request"]
    analysis = result["analysis"]

    exposure = request["exposure"]
    exposure_label = VARIABLE_LABELS.get(
        exposure,
        exposure,
    )

    exposure_type = analysis[
        "exposure_type"
    ]

    unadjusted = analysis[
        "unadjusted"
    ]

    # ========================================================
    # 1. 연속형
    # ========================================================

    if exposure_type == "continuous":

        correlation = float(
            unadjusted["correlation"]
        )

        p_value = float(
            unadjusted["p_value"]
        )

        # 사용자용 핵심 해석
        if p_value < 0.05:

            if correlation > 0:
                summary = (
                    f"{exposure_label} 값이 큰 쪽에서 "
                    "연 소득 5만 달러를 넘는 비율도 "
                    "전반적으로 높은 방향의 관계가 나타났습니다."
                )

            elif correlation < 0:
                summary = (
                    f"{exposure_label} 값이 큰 쪽에서 "
                    "연 소득 5만 달러를 넘는 비율은 "
                    "전반적으로 낮은 방향의 관계가 나타났습니다."
                )

            else:
                summary = (
                    f"{exposure_label}와 연 소득 5만 달러 "
                    "초과 여부 사이에서 뚜렷한 방향은 "
                    "확인되지 않았습니다."
                )

        else:
            summary = (
                f"{exposure_label}와 연 소득 5만 달러 "
                "초과 여부 사이에서 통계적으로 뚜렷한 "
                "관계는 확인되지 않았습니다."
            )

        st.markdown(
            f"**{summary}**"
        )

        st.caption(
            "다른 조건을 따로 고려하지 않고 "
            "현재 데이터에서 두 항목의 관계를 먼저 살펴본 결과입니다."
        )

        with st.expander(
            "분석 상세 정보"
        ):
            detail_left, detail_right = (
                st.columns(2)
            )

            with detail_left:
                st.metric(
                    "상관계수",
                    f"{correlation:.3f}",
                )

            with detail_right:
                st.metric(
                    "p-value",
                    format_p_value(
                        p_value
                    ),
                )

            st.caption(
                "분석 방법 · Point-biserial correlation"
            )


    # ========================================================
    # 2. 이진형
    # ========================================================

    elif exposure_type == "binary":

        metadata = unadjusted[
            "exposure_metadata"
        ]

        reference = metadata[
            "reference_level"
        ]

        comparison = metadata[
            "comparison_level"
        ]

        reference_label = (
            result_value_label(
                exposure,
                reference,
            )
        )

        comparison_label = (
            result_value_label(
                exposure,
                comparison,
            )
        )

        reference_rate = float(
            unadjusted[
                "reference_rate"
            ]
        )

        comparison_rate = float(
            unadjusted[
                "comparison_rate"
            ]
        )

        rate_difference = float(
            unadjusted[
                "rate_difference"
            ]
        )

        # 사용자용 핵심 해석
        if rate_difference > 0:
            summary = (
                f"{comparison_label}에서 연 소득 5만 달러를 "
                f"넘는 비율이 {reference_label}보다 "
                f"{abs(rate_difference) * 100:.1f}%p 높게 나타났습니다."
            )

        elif rate_difference < 0:
            summary = (
                f"{comparison_label}에서 연 소득 5만 달러를 "
                f"넘는 비율이 {reference_label}보다 "
                f"{abs(rate_difference) * 100:.1f}%p 낮게 나타났습니다."
            )

        else:
            summary = (
                f"{reference_label}과 {comparison_label}의 "
                "연 소득 5만 달러 초과 비율은 같게 나타났습니다."
            )

        st.markdown(
            f"**{summary}**"
        )

        st.caption(
            "두 그룹을 다른 조건의 차이를 고려하지 않고 "
            "그대로 비교한 결과입니다."
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:
            st.metric(
                reference_label,
                format_percent(
                    reference_rate
                ),
            )

        with col2:
            st.metric(
                comparison_label,
                format_percent(
                    comparison_rate
                ),
            )

        with col3:
            st.metric(
                "두 그룹의 차이",
                (
                    f"{rate_difference * 100:+.2f}%p"
                ),
            )

        # 기술 통계는 기본 화면에서 숨김
        with st.expander(
            "분석 상세 정보"
        ):

            detail_table = pd.DataFrame(
                [
                    {
                        "통계 항목": "위험비 (Risk Ratio)",
                        "값": unadjusted[
                            "risk_ratio"
                        ],
                    },
                    {
                        "통계 항목": "Odds Ratio",
                        "값": unadjusted[
                            "odds_ratio"
                        ],
                    },
                    {
                        "통계 항목": "Odds Ratio 95% 신뢰구간 하한",
                        "값": unadjusted[
                            "odds_ratio_ci_95_low"
                        ],
                    },
                    {
                        "통계 항목": "Odds Ratio 95% 신뢰구간 상한",
                        "값": unadjusted[
                            "odds_ratio_ci_95_high"
                        ],
                    },
                    {
                        "통계 항목": "Cohen's h",
                        "값": unadjusted[
                            "cohens_h"
                        ],
                    },
                ]
            )

            st.dataframe(
                detail_table,
                width="stretch",
                hide_index=True,
            )

            st.write(
                "Fisher exact test p-value · "
                f"{format_p_value(unadjusted['fisher_exact_p_value'])}"
            )


    # ========================================================
    # 3. 범주형
    # ========================================================

    elif exposure_type == "categorical":

        p_value = float(
            unadjusted[
                "chi2_p_value"
            ]
        )

        groups = pd.DataFrame(
            unadjusted["groups"]
        )

        # 가장 높은 범주 / 가장 낮은 범주
        highest = groups.loc[
            groups["target_rate"].idxmax()
        ]

        lowest = groups.loc[
            groups["target_rate"].idxmin()
        ]

        highest_label = (
            result_value_label(
                exposure,
                highest[exposure],
            )
        )

        lowest_label = (
            result_value_label(
                exposure,
                lowest[exposure],
            )
        )

        highest_rate = float(
            highest["target_rate"]
        )

        lowest_rate = float(
            lowest["target_rate"]
        )

        # 사용자용 핵심 해석
        if p_value < 0.05:
            summary = (
                f"{exposure_label}에 따라 연 소득 5만 달러를 "
                "넘는 비율에 차이가 나타났습니다. "
                f"가장 높은 범주는 {highest_label}"
                f"({format_percent(highest_rate)}), "
                f"가장 낮은 범주는 {lowest_label}"
                f"({format_percent(lowest_rate)})입니다."
            )

        else:
            summary = (
                f"{exposure_label}별 고소득 비율에는 차이가 보이지만, "
                "통계적으로 뚜렷한 차이라고 판단할 근거는 "
                "충분하지 않았습니다."
            )

        st.markdown(
            f"**{summary}**"
        )

        st.caption(
            "각 그룹을 다른 조건의 차이를 고려하지 않고 "
            "그대로 비교한 결과입니다."
        )

        # 사용자용 범주별 표
        user_table = groups.copy()

        user_table[
            exposure
        ] = user_table[
            exposure
        ].map(
            lambda value: (
                result_value_label(
                    exposure,
                    value,
                )
            )
        )

        user_table[
            "연 소득 5만 달러 초과 비율"
        ] = (
            user_table[
                "target_rate"
            ]
            .map(
                lambda value: (
                    f"{value * 100:.2f}%"
                )
            )
        )

        user_table = (
            user_table[
                [
                    exposure,
                    "n",
                    "연 소득 5만 달러 초과 비율",
                ]
            ]
            .rename(
                columns={
                    exposure: exposure_label,
                    "n": "데이터 수",
                }
            )
        )

        st.dataframe(
            user_table,
            width="stretch",
            hide_index=True,
        )

        # 통계 검정은 상세 정보로 이동
        with st.expander(
            "분석 상세 정보"
        ):

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
                    format_p_value(
                        p_value
                    ),
                )

            with col3:
                st.metric(
                    "자유도",
                    unadjusted[
                        "degrees_of_freedom"
                    ],
                )

            expected_cells = (
                unadjusted.get(
                    "expected_cells_under_5",
                    0,
                )
            )

            if expected_cells > 0:
                st.warning(
                    "기대 빈도가 5보다 작은 셀이 있어 "
                    "Chi-square 결과 해석에 주의가 필요합니다."
                )

            st.caption(
                "분석 방법 · Chi-square test"
            )


def display_adjusted_result(
    result: dict,
) -> None:
    """통제 조건을 고려한 결과를 사용자 친화적으로 표시한다."""

    request = result["request"]
    analysis = result["analysis"]
    adjusted = analysis["adjusted"]

    exposure = request["exposure"]
    exposure_type = analysis["exposure_type"]

    exposure_label = VARIABLE_LABELS.get(
        exposure,
        exposure,
    )

    controls = list(
        request.get(
            "controls",
            [],
        )
    )

    effects = adjusted.get(
        "exposure_effects",
        [],
    )

    overall_test = adjusted.get(
        "overall_test"
    )

    # --------------------------------------------------------
    # 적용된 통제 조건 설명
    # --------------------------------------------------------

    if controls:
        control_labels = [
            VARIABLE_LABELS.get(
                control,
                control,
            )
            for control in controls
        ]

        st.caption(
            f"{', '.join(control_labels)}의 차이를 "
            "함께 고려한 결과입니다."
        )

    else:
        st.caption(
            "다른 조건을 추가로 고려하지 않고 "
            "통계 모델로 관계를 다시 확인한 결과입니다."
        )

    # ========================================================
    # 1. 연속형
    # ========================================================

    if exposure_type == "continuous":

        if not effects:
            st.warning(
                "현재 데이터에서는 관계를 "
                "안정적으로 확인하기 어렵습니다."
            )
            return

        effect = effects[0]

        if not effect.get(
            "estimable",
            True,
        ):
            st.warning(
                "현재 데이터에서는 관계를 "
                "안정적으로 확인하기 어렵습니다."
            )
            return

        coefficient = float(
            effect["coefficient"]
        )

        p_value = float(
            effect["p_value"]
        )

        if p_value < 0.05:

            if coefficient > 0:
                summary = (
                    f"{exposure_label} 값이 높을수록 "
                    "연 소득 5만 달러를 넘는 경우가 "
                    "더 많이 나타나는 관계가 확인되었습니다."
                )

            elif coefficient < 0:
                summary = (
                    f"{exposure_label} 값이 높을수록 "
                    "연 소득 5만 달러를 넘는 경우가 "
                    "적게 나타나는 관계가 확인되었습니다."
                )

            else:
                summary = (
                    f"{exposure_label}와 연 소득 5만 달러 "
                    "초과 여부 사이에서 뚜렷한 방향은 "
                    "확인되지 않았습니다."
                )

        else:
            summary = (
                f"{exposure_label}와 연 소득 5만 달러 "
                "초과 여부 사이에서 통계적으로 "
                "뚜렷한 관계는 확인되지 않았습니다."
            )

        st.markdown(
            f"**{summary}**"
        )

        with st.expander(
            "분석 상세 정보"
        ):
            st.metric(
                "p-value",
                format_p_value(
                    p_value
                ),
            )

            st.caption(
                "분석 방법 · Logistic Regression"
            )

    # ========================================================
    # 2. 이진형
    # ========================================================

    elif exposure_type == "binary":

        if not effects:
            st.warning(
                "현재 데이터에서는 두 그룹의 관계를 "
                "안정적으로 확인하기 어렵습니다."
            )
            return

        effect = effects[0]

        if not effect.get(
            "estimable",
            True,
        ):
            st.warning(
                "현재 데이터에서는 두 그룹의 관계를 "
                "안정적으로 확인하기 어렵습니다."
            )
            return

        metadata = adjusted.get(
            "exposure_metadata",
            {},
        )

        reference = metadata.get(
            "reference_level"
        )

        comparison = metadata.get(
            "comparison_level"
        )

        # 앞에서 만든 한글(영어) 표시 함수 재사용
        reference_label = (
            category_value_label(
                exposure,
                reference,
            )
        )

        comparison_label = (
            category_value_label(
                exposure,
                comparison,
            )
        )

        coefficient = float(
            effect["coefficient"]
        )

        p_value = float(
            effect["p_value"]
        )

        if p_value < 0.05:

            if coefficient > 0:
                summary = (
                    f"{comparison_label}에서 "
                    f"{reference_label}보다 연 소득 5만 달러를 "
                    "넘는 경우가 더 많이 나타나는 "
                    "관계가 확인되었습니다."
                )

            elif coefficient < 0:
                summary = (
                    f"{comparison_label}에서 "
                    f"{reference_label}보다 연 소득 5만 달러를 "
                    "넘는 경우가 적게 나타나는 "
                    "관계가 확인되었습니다."
                )

            else:
                summary = (
                    f"{reference_label}과 "
                    f"{comparison_label} 사이에서 "
                    "뚜렷한 차이는 확인되지 않았습니다."
                )

        else:
            summary = (
                f"{reference_label}과 "
                f"{comparison_label} 사이에서 "
                "통계적으로 뚜렷한 차이는 "
                "확인되지 않았습니다."
            )

        st.markdown(
            f"**{summary}**"
        )

        with st.expander(
            "분석 상세 정보"
        ):
            st.metric(
                "p-value",
                format_p_value(
                    p_value
                ),
            )

            st.caption(
                "분석 방법 · Logistic Regression"
            )

    # ========================================================
    # 3. 범주형
    # ========================================================

    elif exposure_type == "categorical":

        if (
            isinstance(
                overall_test,
                dict,
            )
            and overall_test.get(
                "estimable",
                False,
            )
        ):

            p_value = float(
                overall_test["p_value"]
            )

            if p_value < 0.05:
                summary = (
                    f"{exposure_label}에 따라 "
                    "연 소득 5만 달러 초과 여부에 "
                    "통계적으로 뚜렷한 차이가 "
                    "확인되었습니다."
                )

            else:
                summary = (
                    f"{exposure_label} 전체를 살펴봤을 때 "
                    "연 소득 5만 달러 초과 여부와의 "
                    "통계적으로 뚜렷한 관계는 "
                    "확인되지 않았습니다."
                )

            st.markdown(
                f"**{summary}**"
            )

            with st.expander(
                "분석 상세 정보"
            ):
                detail_left, detail_right = (
                    st.columns(2)
                )

                with detail_left:
                    st.metric(
                        "p-value",
                        format_p_value(
                            p_value
                        ),
                    )

                with detail_right:
                    st.metric(
                        "자유도",
                        overall_test[
                            "degrees_of_freedom"
                        ],
                    )

                st.caption(
                    "분석 방법 · Logistic Regression "
                    "전체 범주 검정"
                )

        else:
            st.warning(
                "현재 데이터에서는 이 항목 전체의 관계를 "
                "안정적으로 확인하기 어렵습니다."
            )

    # ========================================================
    # 공통 추정 경고
    # ========================================================

    warnings = adjusted.get(
        "estimation_warnings",
        [],
    )

    if warnings:
        unstable_labels = []

        for effect in effects:
            if effect.get(
                "estimable",
                True,
            ):
                continue

            term = str(
                effect.get(
                    "term",
                    "",
                )
            )

            prefix = f"{exposure}_"

            if term.startswith(
                prefix
            ):
                raw_value = term[
                    len(prefix):
                ]
                label = (
                    CATEGORY_VALUE_LABELS
                    .get(exposure, {})
                    .get(raw_value, raw_value)
                )
                unstable_labels.append(label)

        if unstable_labels:
            st.warning(
                "일부 범주는 데이터 수가 적거나 "
                "결과가 한쪽에 치우쳐 있어 "
                "개별 효과를 안정적으로 계산하기 어렵습니다. "
                f"해당 범주: {', '.join(unstable_labels)}"
            )
        else:
            st.warning(
                "일부 범주는 데이터가 충분하지 않아 "
                "개별 효과를 안정적으로 계산하기 어렵습니다."
            )


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
            format_p_value(
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


def association_page(
    df: pd.DataFrame,
) -> None:
    """사용자 선택형 고소득 연관성 분석 UI."""

    st.header(
        "소득과의 관계"
    )

    st.markdown(
        """
        <div class="section-description">
            궁금한 항목을 하나 골라 연 소득 5만 달러를 넘는 경우와
            어떤 관계가 있는지 살펴보세요.<br>
            다른 조건도 함께 선택하면 그 차이까지 고려해서 비교할 수 있어요.
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
                "무엇을 살펴볼까요?"
            )

            exposure = st.selectbox(
                "궁금한 항목",
                options=list(
                    ANALYSIS_VARIABLES
                ),
                format_func=lambda variable: (
                    VARIABLE_LABELS.get(
                        variable,
                        variable,
                    )
                ),
                key="association_exposure",
            )

            exposure_type = (
                ANALYSIS_VARIABLE_TYPES[
                    exposure
                ]
            )

            control_options = [
                variable
                for variable
                in ANALYSIS_VARIABLES
                if variable != exposure
            ]

            controls = st.multiselect(
                "함께 고려할 항목",
                options=control_options,
                format_func=lambda variable: (
                    VARIABLE_LABELS.get(
                        variable,
                        variable,
                    )
                ),
                key="association_controls",
                placeholder=(
                    "추가로 고려할 항목을 골라주세요"
                ),
            )

            st.caption(
                "선택하지 않아도 돼요."
            )

            psm_available = (
                exposure_type == "binary"
                and bool(
                    controls
                )
            )

            if psm_available:
                include_psm = st.toggle(
                    "비슷한 조건끼리 추가로 비교하기",
                    value=False,
                    help=(
                        "선택한 조건이 비슷한 사람끼리 묶어서 "
                        "차이를 한 번 더 비교합니다."
                    ),
                )

            else:
                include_psm = False

            analyze_button = st.button(
                "결과 보기",
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

        # --------------------------------------------------------
        # 사용자에게 보여줄 질문
        # --------------------------------------------------------

        if control_labels:
            controls_text = ", ".join(
                control_labels
            )

            question = (
                f"{controls_text}도 함께 고려했을 때, "
                f"{exposure_label}에 따라 연 소득 5만 달러를 "
                "넘을 가능성이 달라질까요?"
            )

            description = (
                "선택한 조건들의 차이를 함께 고려해서 "
                f"{exposure_label}와 소득의 관계를 살펴봐요."
            )

        else:
            if exposure_type == "continuous":
                question = (
                    f"{exposure_label}에 따라 연 소득 5만 달러를 "
                    "넘는 비율이 어떻게 달라질까요?"
                )

                description = (
                    "실제 데이터에서 값이 달라질수록 "
                    "소득 수준에도 차이가 나타나는지 살펴봐요."
                )

            elif exposure_type == "binary":
                question = (
                    f"{exposure_label}에 따라 연 소득 5만 달러를 "
                    "넘는 비율이 다를까요?"
                )

                description = (
                    "두 그룹에서 연 소득 5만 달러를 넘는 "
                    "비율에 차이가 있는지 비교해요."
                )

            else:
                question = (
                    f"{exposure_label}에 따라 연 소득 5만 달러를 "
                    "넘는 비율이 어떻게 다를까요?"
                )

                description = (
                    "각 그룹에서 연 소득 5만 달러를 넘는 "
                    "비율이 어떻게 다른지 비교해요."
                )

        with st.container(
            border=True
        ):
            st.markdown(
                """
                <div class="section-number">
                    지금 살펴볼 내용
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"### {question}"
            )

            st.write("")

            st.caption(
                description
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
            display_association_error(
                exc
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

    request_result = result["request"]
    analysis = result["analysis"]

    # 마지막으로 '결과 보기'를 눌렀을 때 실제 적용된 조건
    applied_controls = list(
        request_result.get(
            "controls",
            [],
        )
    )

    analysis_rows = int(
        analysis["sample_size"]
    )

    excluded_rows = int(
        analysis[
            "rows_excluded_due_to_missing"
        ]
    )

    control_text = (
        "없음"
        if not applied_controls
        else f"{len(applied_controls)}개"
    )


    # 결과 영역 상단 여백
    st.markdown(
        '<div class="result-top-spacer"></div>',
        unsafe_allow_html=True,
    )


    if excluded_rows == 0:
        metric_left, metric_right = st.columns(2)

        with metric_left:
            st.metric(
                "분석에 사용한 데이터",
                f"{analysis_rows:,}명",
                border=True,
            )

        with metric_right:
            st.metric(
                "함께 고려한 항목",
                control_text,
                border=True,
            )

    else:
        metric_left, metric_center, metric_right = (
            st.columns(3)
        )

        with metric_left:
            st.metric(
                "분석에 사용한 데이터",
                f"{analysis_rows:,}명",
                border=True,
            )

        with metric_center:
            st.metric(
                "제외된 데이터",
                f"{excluded_rows:,}명",
                border=True,
            )

        with metric_right:
            st.metric(
                "함께 고려한 항목",
                control_text,
                border=True,
            )
    
    # --------------------------------------------------------
    # 조정 전
    # --------------------------------------------------------

    executed_exposure = request_result[
        "exposure"
    ]

    executed_exposure_type = analysis[
        "exposure_type"
    ]

    executed_exposure_label = (
        VARIABLE_LABELS.get(
            executed_exposure,
            executed_exposure,
        )
    )

    st.markdown(
        '<div class="result-section-spacer"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-number">
            01 · 데이터에서 보이는 관계
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader(
        f"{executed_exposure_label}와 소득의 관계"
    )

    display_unadjusted_result(
        result
    )

    # --------------------------------------------------------
    # 조정 후
    # --------------------------------------------------------

    # --------------------------------------------------------
    # 다른 조건을 함께 고려한 결과
    # --------------------------------------------------------

    applied_exposure = (
        request_result[
            "exposure"
        ]
    )

    applied_controls = list(
        request_result.get(
            "controls",
            [],
        )
    )

    applied_exposure_label = (
        VARIABLE_LABELS.get(
            applied_exposure,
            applied_exposure,
        )
    )

    st.markdown(
        '<div class="result-section-spacer"></div>',
        unsafe_allow_html=True,
    )

    if applied_controls:

        st.markdown(
            """
            <div class="section-number">
                02 · 다른 조건을 함께 고려한 결과
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader(
            f"{applied_exposure_label}와 소득의 관계가 "
            "다른 조건을 고려해도 나타날까요?"
        )

    else:

        st.markdown(
            """
            <div class="section-number">
                02 · 통계 모델로 다시 확인한 결과
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader(
            f"{applied_exposure_label}와 소득의 관계를 "
            "모델로 다시 확인해봤어요."
        )


    display_adjusted_result(
        result
    )

# --------------------------------------------------------
# 예상 비율
# --------------------------------------------------------

    applied_exposure = (
        request_result[
            "exposure"
        ]
    )

    applied_controls = list(
        request_result.get(
            "controls",
            [],
        )
    )

    applied_exposure_label = (
        VARIABLE_LABELS.get(
            applied_exposure,
            applied_exposure,
        )
    )

    st.markdown(
        '<div class="result-section-spacer"></div>',
        unsafe_allow_html=True,
    )

    if applied_controls:

        st.markdown(
            """
            <div class="section-number">
                03 · 조건을 고려한 예상 비율
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="section-number">
                03 · 모델이 계산한 예상 비율
            </div>
            """,
            unsafe_allow_html=True,
        )


    st.subheader(
        f"{applied_exposure_label}에 따라 "
        "예상 비율이 어떻게 달라질까요?"
    )


    if applied_controls:

        control_labels = [
            VARIABLE_LABELS.get(
                control,
                control,
            )
            for control
            in applied_controls
        ]

        controls_text = ", ".join(
            control_labels
        )

        st.caption(
            f"{controls_text}의 차이를 함께 고려했을 때, "
            f"{applied_exposure_label}에 따른 "
            "연 소득 5만 달러 초과 예상 비율을 보여줍니다."
        )

    else:

        st.caption(
            "다른 조건을 추가로 고려하지 않은 상태에서, "
            f"{applied_exposure_label}에 따른 "
            "연 소득 5만 달러 초과 예상 비율을 보여줍니다."
        )


    if (
        "adjusted_probability"
        in figures
    ):
        display_plotly_chart(
            figures[
                "adjusted_probability"
            ]
        )

    # --------------------------------------------------------
    # PSM
    # --------------------------------------------------------

    psm = analysis.get(
        "psm"
    )

    st.markdown(
        '<div class="result-section-spacer"></div>',
        unsafe_allow_html=True,
    )
    
    if psm is not None:
        st.markdown(
            '<div class="section-number">'
            '04 · PROPENSITY SCORE MATCHING'
            '</div>',
            unsafe_allow_html=True,
        )

        st.subheader(
            "4. 성향점수매칭(PSM)"
        )

        display_psm_result(
            psm
        )

        if (
            "psm_balance"
            in figures
        ):
            display_plotly_chart(
                figures[
                    "psm_balance"
                ]
            )

        display_interpretation_note(
            psm[
                "result"
            ][
                "interpretation_note"
            ]
        )

    st.markdown(
        "#### 결과를 볼 때 참고해주세요"
    )

    if request_result.get(
        "include_psm",
        False,
    ):
        st.caption(
            "비슷한 조건의 사람끼리 추가로 비교했지만, "
            "데이터에 포함되지 않은 다른 차이까지 모두 고려할 수는 없습니다. "
            "따라서 이 결과를 직접적인 원인과 결과로 해석해서는 안 됩니다."
        )

    else:
        st.caption(
            "이 결과는 데이터에서 함께 나타나는 관계를 보여줍니다. "
            "다른 조건을 함께 고려했더라도 특정 항목이 "
            "소득 차이의 직접적인 원인이라고 단정할 수는 없습니다."
        )


def display_association_error(
    exc: Exception,
) -> None:
    """연관성 분석 실패 원인을 사용자 친화적으로 안내한다."""

    logger.exception(
        "연관성 분석 실행 실패"
    )
    message = str(exc)

    if (
        "수렴하지 않았습니다"
        in message
        or "행렬 계산에 실패했습니다"
        in message
        or "선형 종속성"
        in message
        or "완전히 분리"
        in message
    ):
        st.warning(
            "현재 변수 조합에서는 조정된 연관성을 "
            "안정적으로 계산하기 어렵습니다."
        )

        st.markdown(
            """
            **다음 방법 중 하나를 시도해 주세요.**

            - 통제 변수를 1개 이상 줄여 보세요.
            - 범주가 많은 통제 변수(예: 직업, 출신 국가)를 제외해 보세요.
            - 다른 관심 변수 또는 통제 변수 조합을 선택해 보세요.

            일부 범주의 표본이 매우 적거나 고소득 여부가 한쪽으로
            치우치면 Logistic Regression의 Odds Ratio를 안정적으로
            추정하지 못할 수 있습니다.
            """
        )

        st.caption(
            "같은 문제가 반복되면 페이지 하단의 문의하기에서 "
            "오류 신고로 알려주세요."
        )
        return

    st.error(
        "분석을 완료하지 못했습니다. "
        "변수 선택을 확인한 뒤 다시 시도해 주세요."
    )
    st.caption(
        "문제가 반복되면 페이지 하단의 문의하기에서 "
        "오류 신고로 알려주세요."
    )
