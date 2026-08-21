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
    plot_what_if_simulation,
)


# ============================================================
# 페이지 설정
# ============================================================

st.set_page_config(
    page_title="소득 데이터 탐색 | SKALA",
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

    h2 { margin-top: 1.8rem !important;}

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

    .hero-title, h1, h2 {
        font-family:
            "Cormorant Garamond",
            "Noto Serif KR",
            Georgia,
            serif;
    }

    body, p, label, button, input, textarea, [data-testid="stMetric"] {
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

    .result-top-spacer {
        height: 1.25rem;
    }

    .result-section-spacer {
        height: 2rem;
    }
    /* -------------------------------------------------------
       Streamlit components
    ------------------------------------------------------- */

    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.35);
        border-radius: 10px;
    }

    div[data-testid="stMetric"] {background: rgba(255, 255, 255, 0.45);}
    div[data-baseweb="select"] > div {background-color: #FCFBF8;}
    div[data-testid="stVerticalBlock"] {gap: 0.8rem;}

    hr {
        margin-top: 1.2rem !important;
        margin-bottom: 1.6rem !important;
    }

    hr {border-color: #DDDAD3 !important;}
    /* -------------------------------------------------------
    Streamlit chrome
    ------------------------------------------------------- */

    header[data-testid="stHeader"] {
        display: none;
    }

    [data-testid="stToolbar"] {
        display: none;
    }

    #MainMenu {
        visibility: hidden;
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
    "workclass": "고용 형태",
    "education": "교육 수준",
    "marital-status": "혼인 상태",
    "occupation": "직업",
    "relationship": "가구 내 관계",
    "race": "인종",
    "sex": "성별",
    "capital-gain": "투자·자산 이익",
    "capital-loss": "투자·자산 손실",
    "hours-per-week": "주당 근무시간",
    "native-country": "출신 국가",
}

CATEGORY_VALUE_LABELS = {
    "sex": {
        "Male": "남성",
        "Female": "여성",
    },

    "race": {
        "White": "백인",
        "Black": "흑인",
        "Asian-Pac-Islander": "아시아·태평양계",
        "Amer-Indian-Eskimo": "아메리카 원주민·알래스카 원주민",
        "Other": "기타",
    },

    "relationship": {
        "Husband": "남편",
        "Wife": "아내",
        "Own-child": "자녀",
        "Not-in-family": "가족 외",
        "Other-relative": "기타 친족",
        "Unmarried": "미혼·비혼",
    },

    "marital-status": {
        "Married-civ-spouse": "기혼·배우자 동거",
        "Divorced": "이혼",
        "Never-married": "미혼",
        "Separated": "별거",
        "Widowed": "사별",
        "Married-spouse-absent": "기혼·배우자 부재",
        "Married-AF-spouse": "군인 배우자와 기혼",
    },

    "workclass": {
        "Private": "민간 기업",
        "Self-emp-not-inc": "자영업·비법인",
        "Self-emp-inc": "자영업·법인",
        "Federal-gov": "연방정부",
        "Local-gov": "지방정부",
        "State-gov": "주정부",
        "Without-pay": "무급 근무",
        "Never-worked": "근무 경험 없음",
    },

    "education": {
        "Preschool": "취학 전",
        "1st-4th": "초등 1~4학년",
        "5th-6th": "초등 5~6학년",
        "7th-8th": "중학교 수준",
        "9th": "9학년",
        "10th": "10학년",
        "11th": "11학년",
        "12th": "12학년",
        "HS-grad": "고등학교 졸업",
        "Some-college": "대학 일부 이수",
        "Assoc-voc": "전문학사·직업 과정",
        "Assoc-acdm": "전문학사·학술 과정",
        "Bachelors": "학사",
        "Masters": "석사",
        "Prof-school": "전문대학원",
        "Doctorate": "박사",
    },

    "occupation": {
        "Adm-clerical": "사무·행정직",
        "Armed-Forces": "군인",
        "Craft-repair": "기능·수리직",
        "Exec-managerial": "관리·경영직",
        "Farming-fishing": "농림·어업",
        "Handlers-cleaners": "운반·청소직",
        "Machine-op-inspct": "기계 조작·검사직",
        "Other-service": "기타 서비스직",
        "Priv-house-serv": "가사 서비스직",
        "Prof-specialty": "전문직",
        "Protective-serv": "보안·보호 서비스직",
        "Sales": "판매직",
        "Tech-support": "기술 지원직",
        "Transport-moving": "운송직",
    },

    "native-country": {
        "United-States": "미국",
        "Canada": "캐나다",
        "Mexico": "멕시코",
        "Puerto-Rico": "푸에르토리코",
        "Cuba": "쿠바",
        "Jamaica": "자메이카",
        "Dominican-Republic": "도미니카공화국",
        "Haiti": "아이티",
        "Guatemala": "과테말라",
        "Honduras": "온두라스",
        "Nicaragua": "니카라과",
        "El-Salvador": "엘살바도르",
        "Trinadad&Tobago": "트리니다드 토바고",

        "England": "영국",
        "Germany": "독일",
        "France": "프랑스",
        "Italy": "이탈리아",
        "Poland": "폴란드",
        "Portugal": "포르투갈",
        "Ireland": "아일랜드",
        "Greece": "그리스",
        "Hungary": "헝가리",
        "Scotland": "스코틀랜드",
        "Yugoslavia": "유고슬라비아",
        "Holand-Netherlands": "네덜란드",

        "India": "인도",
        "China": "중국",
        "Japan": "일본",
        "Vietnam": "베트남",
        "Philippines": "필리핀",
        "Thailand": "태국",
        "Cambodia": "캄보디아",
        "Laos": "라오스",
        "Taiwan": "대만",
        "Hong": "홍콩",

        "Iran": "이란",

        "Columbia": "콜롬비아",
        "Ecuador": "에콰도르",
        "Peru": "페루",

        "South": "대한민국",
        "Outlying-US(Guam-USVI-etc)": "미국령 지역",
    },
}

VARIABLE_TYPE_LABELS = {
    "binary": "이진형",
    "continuous": "연속형",
    "categorical": "범주형",
}

def _category_value_label(
    feature: str,
    value: str,
) -> str:
    """범주 값을 한글(영어) 형식으로 표시한다."""

    korean = (
        CATEGORY_VALUE_LABELS
        .get(feature, {})
        .get(value)
    )

    if korean is None:
        return str(value)

    return f"{korean} ({value})"

def _variable_label(
    variable: str,
) -> str:
    """변수명을 사용자 표시용 이름으로 변환한다."""

    korean = VARIABLE_LABELS.get(
        variable,
        variable,
    )

    return (f"{korean} ({variable})")


# ============================================================
# 데이터 및 모델 메타데이터 캐시
# ============================================================

@st.cache_data
def load_service_data() -> pd.DataFrame:
    """서비스에서 공통으로 사용할 정제 Adult 데이터를 로딩한다."""

    return load_and_clean(save_output=False,)

@st.cache_data
def load_prediction_schema() -> dict:
    """예측 입력 폼 생성에 필요한 모델 스키마를 반환한다."""

    return (get_prediction_input_schema())

@st.cache_data
def load_global_importance() -> pd.DataFrame:
    """현재 저장된 모델의 전체 permutation importance를 반환한다."""

    return (get_global_feature_importance())

# ============================================================
# 공통 표시 함수
# ============================================================

def _format_p_value(
    value: float | None,
) -> str:
    """p-value를 화면에 적절한 문자열로 표시한다."""

    if value is None:
        return "추정 불가"

    value = float(value)

    if value < 0.001:
        return f"{value:.2e}"

    return f"{value:.4f}"


def _format_percent(
    value: float | None,
) -> str:
    """0~1 비율을 백분율 문자열로 변환한다."""

    if value is None:
        return "-"

    return (f"{float(value) * 100:.2f}%")


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

def _result_value_label(
    variable: str,
    value,
) -> str:
    """분석 결과의 범주 값을 사용자 표시용 이름으로 변환한다."""

    labels = globals().get(
        "CATEGORY_VALUE_LABELS",
        {},
    )

    korean = (
        labels
        .get(variable, {})
        .get(value)
    )

    if korean is None:
        return str(value)

    return f"{korean} ({value})"

# ============================================================
# 조정 전 결과
# ============================================================

# ============================================================
# 데이터에서 보이는 관계
# ============================================================

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
                    _format_p_value(
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
            _result_value_label(
                exposure,
                reference,
            )
        )

        comparison_label = (
            _result_value_label(
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
                _format_percent(
                    reference_rate
                ),
            )

        with col2:
            st.metric(
                comparison_label,
                _format_percent(
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
                f"{_format_p_value(unadjusted['fisher_exact_p_value'])}"
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
            _result_value_label(
                exposure,
                highest[exposure],
            )
        )

        lowest_label = (
            _result_value_label(
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
                f"({_format_percent(highest_rate)}), "
                f"가장 낮은 범주는 {lowest_label}"
                f"({_format_percent(lowest_rate)})입니다."
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
                _result_value_label(
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
                    _format_p_value(
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

# ============================================================
# Logistic Regression 결과
# ============================================================

# ============================================================
# 다른 조건을 함께 고려한 결과
# ============================================================

# ============================================================
# 다른 조건을 함께 고려한 결과
# ============================================================

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
                _format_p_value(
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
            _category_value_label(
                exposure,
                reference,
            )
        )

        comparison_label = (
            _category_value_label(
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
                _format_p_value(
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
                        _format_p_value(
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

    for warning in warnings:
        st.warning(
            warning
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
        st.plotly_chart(
            figures[
                "adjusted_probability"
            ],
            width="stretch",
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
        VARIABLE_LABELS.get(
            feature,
            feature,
        ),
        options=levels,
        index=default_index,
        format_func=lambda value: (
            _category_value_label(
                feature,
                value,
            )
        ),
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

        except VisualizationError as exc:
            st.warning(
                f"입력값 비교 그래프를 표시하지 못했습니다: {exc}"
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
    ) as exc:
        st.warning(
            f"예측 중요도를 표시하지 못했습니다: {exc}"
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

# ============================================================
# 앱 실행
# ============================================================

def display_association_error(
    exc: Exception,
) -> None:
    """연관성 분석 실패 원인을 사용자 친화적으로 안내한다."""

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

        with st.expander("기술적 상세 정보"):
            st.code(message)

        return

    st.error(message)

def render_hero() -> None:
    """서비스의 상단 소개 영역을 표시한다."""

    st.markdown(
        """
        <div class="hero-eyebrow">
            SKALA · 소득 데이터 탐색
        </div>

        <div class="hero-title">
            소득이 높은 사람들은 어떤 점이 다를까요?
        </div>

        <div class="hero-description">
            나이, 교육 수준, 직업 같은 조건에 따라
            소득에 어떤 차이가 있는지 살펴보고,<br>
            내 조건에서는 연 소득 5만 달러(약 7500만원)를 넘을 가능성이
            얼마나 되는지도 확인해보세요.
        </div>
        """,
        unsafe_allow_html=True,
    )

def main() -> None:
    """Streamlit 웹 애플리케이션을 실행한다."""

    render_hero()

    try:
        df = (load_service_data())
    except Exception as exc:
        st.error(
            "Adult 데이터를 불러오지 못했습니다: "
            f"{exc}"
        )
        st.stop()

    # ============================================================
    # 페이지 네비게이션
    # ============================================================

    if "service_mode" not in st.session_state:
        st.session_state[
            "service_mode"
        ] = "소득과의 관계"


    nav_left, nav_center, nav_right = st.columns(
        [1, 3, 1]
    )

    with nav_center:
        tab_left, tab_right = st.columns(
            2,
            gap="small",
        )

        current_mode = st.session_state[
            "service_mode"
        ]

        with tab_left:
            association_clicked = st.button(
                "소득과의 관계",
                key="nav_association",
                type=(
                    "primary"
                    if current_mode == "소득과의 관계"
                    else "secondary"
                ),
                width="stretch",
            )

        with tab_right:
            prediction_clicked = st.button(
                "내 소득 예측",
                key="nav_prediction",
                type=(
                    "primary"
                    if current_mode == "내 소득 예측"
                    else "secondary"
                ),
                width="stretch",
            )

        if association_clicked:
            st.session_state[
                "service_mode"
            ] = "소득과의 관계"
            st.rerun()

        if prediction_clicked:
            st.session_state[
                "service_mode"
            ] = "내 소득 예측"
            st.rerun()


    st.divider()


    if (
        st.session_state[
            "service_mode"
        ]
        == "소득과의 관계"
    ):
        association_page(
            df
        )

    else:
        prediction_page()

if __name__ == "__main__":
    main()