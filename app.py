"""Adult Census Income 연관성 분석·예측 웹 애플리케이션.

사용자 기능:
    1. 관심 변수와 통제 변수를 선택한 고소득 연관성 분석
    2. 사용자 입력 조건에 대한 고소득 확률 예측
    3. 개인 예측 설명
    4. What-if 시뮬레이션
    5. 전체 모델 기준 Feature Importance 확인
    6. 사용자 문의 접수

분석·예측 계산과 각 페이지 UI는 src 모듈이 담당하며,
이 파일은 앱 진입점, 공통 스타일, 데이터 로딩, 네비게이션을 담당한다.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.data import load_and_clean
from src.ui_association import association_page
from src.ui_inquiry import render_inquiry_section
from src.ui_prediction import prediction_page
logger = logging.getLogger(__name__)


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
# 서비스 데이터 캐시
# ============================================================

@st.cache_data
def load_service_data() -> pd.DataFrame:
    """서비스에서 공통으로 사용할 정제 Adult 데이터를 로딩한다."""

    return load_and_clean(
        save_output=False,
    )

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
    except Exception:
        logger.exception(
            "서비스 데이터 로드 실패"
        )
        st.error(
            "서비스 데이터를 불러오지 못했습니다. "
            "잠시 후 다시 시도해 주세요."
        )
        st.caption(
            "문제가 계속되면 관리자에게 문의해 주세요."
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

    render_inquiry_section(
        current_page=st.session_state[
            "service_mode"
        ],
    )

if __name__ == "__main__":
    main()
