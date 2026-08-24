"""개인 소득 예측 Streamlit UI.

모델 계산은 src.modeling이 담당하고,
이 모듈은 예측 입력·결과 설명·What-if 화면만 담당한다.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.labels import (
    CATEGORY_VALUE_LABELS,
    VARIABLE_LABELS,
)
from src.modeling import (
    ModelingError,
    get_global_feature_importance,
    get_prediction_input_schema,
    predict_income_input,
    simulate_income_what_if,
)
from src.ui_common import (
    category_value_label,
    variable_label,
)
from src.visualization import (
    VisualizationError,
    plot_global_feature_importance,
    plot_prediction_explanation,
    plot_what_if_simulation,
)


logger = logging.getLogger(__name__)


@st.cache_data
def load_prediction_schema() -> dict:
    """예측 입력 폼 생성에 필요한 모델 스키마를 반환한다."""

    return (get_prediction_input_schema())


@st.cache_data
def load_global_importance() -> pd.DataFrame:
    """현재 저장된 모델의 전체 permutation importance를 반환한다."""

    return (get_global_feature_importance())


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
                variable_label(
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
            variable_label(
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
        VARIABLE_LABELS.get(
            feature,
            feature,
        ),
        options=levels,
        index=default_index,
        format_func=lambda value: (
            category_value_label(
                feature,
                value,
            )
        ),
        key=(
            f"prediction_{feature}"
        ),
    )


def prediction_page() -> None:
    """개별 입력의 고소득 예측과 모델 설명 UI."""

    st.header(
        "내 소득 예측"
    )

    st.write(
        "내 정보를 입력하면 현재 모델이 "
        "연 소득 5만 달러를 넘을 가능성을 보여드려요."
    )

    st.caption(
        "입력한 정보는 예측에만 사용되며, "
        "앞에서 살펴본 '소득과의 관계' 분석과는 별도로 계산됩니다."
    )

    try:
        schema = (
            load_prediction_schema()
        )
    except ModelingError:
        logger.exception(
            "예측 입력 스키마 로드 실패"
        )
        st.error(
            "예측 서비스를 준비하지 못했습니다. "
            "잠시 후 다시 시도해 주세요."
        )
        st.caption(
            "문제가 계속되면 페이지 하단의 문의하기에서 "
            "오류 신고로 알려주세요."
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
                '<div class="result-section-spacer"></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="section-number">
                    01 · 기본 정보
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader(
                "나에 대한 정보"
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
                    02 · 학력과 일
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader(
                "학력과 직업 정보"
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
                '<div class="result-section-spacer"></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="section-number">
                    03 · 추가 소득 정보
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader(
                "투자·자산 거래에서 생긴 이익과 손실"
            )

            st.caption(
                "월급이나 현재 보유한 자산을 입력하는 항목이 아닙니다. "
                "해당하는 이익이나 손실이 없다면 0으로 입력하세요."
            )

            left, right = (
                st.columns(2)
            )

            with left:
                render_prediction_feature(
                    "capital-gain"
                )

                st.caption(
                    "주식이나 자산 거래 등에서 발생한 이익입니다. "
                    "해당 사항이 없다면 0으로 입력하세요."
                )

            with right:
                render_prediction_feature(
                    "capital-loss"
                )

                st.caption(
                    "주식이나 자산 거래 등에서 발생한 손실입니다. "
                    "해당 사항이 없다면 0으로 입력하세요."
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
                    "결과 확인",
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

        except ModelingError:
            logger.exception(
                "개인 소득 예측 실행 실패"
            )
            st.error(
                "예측 결과를 계산하지 못했습니다. "
                "입력값을 확인한 뒤 다시 시도해 주세요."
            )
            st.caption(
                "같은 문제가 반복되면 페이지 하단의 문의하기에서 "
                "오류 신고로 알려주세요."
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
        # 새로운 예측을 실행하면
        # 이전 What-if 결과를 초기화한다.
        for key in [
            "what_if_result",
            "what_if_figure",
            "what_if_result_feature",
        ]:
            st.session_state.pop(
                key,
                None,
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

    probability = float(
        prediction[
            "high_income_probability"
        ]
    )

    st.markdown(
        """
        <div class="section-number">
            01 · 내 예측 결과
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader(
        "연 소득 5만 달러를 넘을 가능성"
    )

    st.metric(
        label="예측 확률",
        value=f"{probability * 100:.1f}%",
    )

    if probability >= 0.5:
        st.markdown(
            "**현재 입력한 조건에서는 연 소득 5만 달러를 "
            "넘을 가능성이 조금 더 높게 예측됐습니다.**"
        )

    else:
        st.markdown(
            "**현재 입력한 조건에서는 연 소득 5만 달러 이하일 "
            "가능성이 조금 더 높게 예측됐습니다.**"
        )

    st.caption(
        "입력한 정보를 바탕으로 모델이 계산한 예측값이며, "
        "실제 소득을 의미하지 않습니다."
    )

    # --------------------------------------------------------
    # 내 입력값에 따른 예측 변화
    # --------------------------------------------------------
    st.markdown(
        '<div class="result-section-spacer"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-number">
            02 · 내 입력값 살펴보기
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader(
        "어떤 입력값에서 예측이 달라졌을까요?"
    )

    st.caption(
        "현재 입력한 값을 학습 데이터에서 자주 나타나는 대표적인 값으로 "
        "하나씩 바꿔보면서 예측 확률이 얼마나 달라지는지 비교합니다."
    )

    explanation_features = (
        prediction_result
        .get(
            "explanation",
            {},
        )
        .get(
            "features",
            [],
        )
    )

    max_impact = max(
        (
            abs(
                float(
                    item.get(
                        "impact_percentage_points",
                        0,
                    )
                )
            )
            for item in explanation_features
        ),
        default=0.0,
    )


    # 변화가 사실상 없는 경우
    if max_impact < 0.01:

        st.info(
            "현재 입력에서는 각 항목을 대표적인 값으로 바꿔도 "
            "예측 확률의 변화가 거의 없었습니다."
        )

        st.caption(
            "현재 입력값이 학습 데이터의 대표적인 값과 같거나 비슷하면 "
            "이런 결과가 나타날 수 있습니다."
        )


    # 의미 있는 변화가 있는 경우에만 그래프 표시
    else:

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

        except VisualizationError:
            logger.exception(
                "개인 예측 설명 그래프 생성 실패"
            )
            st.warning(
                "입력값 비교 그래프를 표시하지 못했습니다. "
                "예측 결과 자체는 그대로 확인할 수 있습니다."
            )

        st.caption(
            "오른쪽으로 갈수록 현재 입력값에서 예측 확률이 더 높았고, "
            "왼쪽으로 갈수록 더 낮았습니다. "
            "각 항목은 하나씩 따로 바꿔본 결과이며 "
            "원인과 결과를 의미하지 않습니다."
        )

    # --------------------------------------------------------
    # 전체 모델 기준 중요도
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-number">
            03 · 모델이 많이 참고한 정보
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader(
        "모델은 어떤 정보를 많이 참고했을까요?"
    )

    st.caption(
        "전체 데이터를 기준으로 봤을 때 "
        "모델이 예측에 많이 활용한 항목을 보여줍니다."
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
    ):
        logger.exception(
            "전체 모델 중요도 표시 실패"
        )
        st.warning(
            "모델 중요도 그래프를 표시하지 못했습니다. "
            "개인 예측 결과는 그대로 사용할 수 있습니다."
        )

    st.caption(
        "값이 클수록 해당 항목을 섞었을 때 "
        "모델의 예측 성능이 더 많이 떨어졌다는 뜻입니다. "
        "개인별 예측 결과나 실제 소득에 미치는 영향의 크기를 의미하지 않습니다."
    )

    # --------------------------------------------------------
    # 조건을 바꿔서 확인하기
    # --------------------------------------------------------
    st.markdown(
        '<div class="result-section-spacer"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-number">
            04 · what-if 
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader(
        "What-if simulation"
    )

    st.caption(
        "나머지 정보는 그대로 두고, "
        "선택한 항목만 바꿨을 때 "
        "연 소득 5만 달러 초과 예측 확률이 "
        "어떻게 달라지는지 확인해보세요."
    )

    what_if_feature = st.selectbox(
        "바꿔볼 항목",
        options=feature_columns,
        format_func=lambda feature: (
            VARIABLE_LABELS.get(
                feature,
                feature,
            )
        ),
        key="what_if_feature",
    )

    if st.button(
        "변화 확인하기",
        type="primary",
        width="stretch",
    ):
        try:
            with st.spinner(
                "조건을 바꿨을 때의 결과를 계산하고 있습니다..."
            ):
                what_if = (
                    simulate_income_what_if(
                        prediction_input,
                        feature=what_if_feature,
                    )
                )

                what_if_figure = (
                    plot_what_if_simulation(
                        what_if,
                        feature_label=(
                            VARIABLE_LABELS.get(
                                what_if_feature,
                                what_if_feature,
                            )
                        ),
                        category_labels=(
                            CATEGORY_VALUE_LABELS.get(
                                what_if_feature,
                                {},
                            )
                        ),
                        current_value=(
                            prediction_input.get(
                                what_if_feature
                            )
                        ),
                    )
                )

        except (
            ModelingError,
            VisualizationError,
        ):
            logger.exception(
                "What-if 시뮬레이션 실행 실패"
            )
            st.error(
                "조건 변경 결과를 계산하지 못했습니다. "
                "다른 항목을 선택한 뒤 다시 시도해 주세요."
            )
            st.caption(
                "문제가 반복되면 페이지 하단의 문의하기에서 "
                "오류 신고로 알려주세요."
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


    what_if_figure = st.session_state.get(
        "what_if_figure"
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
            "다른 조건은 그대로 둔 채 한 항목만 바꿔본 "
            "모델의 예측 결과입니다. "
            "실제로 해당 조건을 바꾸면 소득이 이렇게 "
            "변한다는 뜻은 아닙니다."
        )
