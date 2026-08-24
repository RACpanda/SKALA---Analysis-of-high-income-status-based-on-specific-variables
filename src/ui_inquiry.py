"""사용자 문의 기능의 Streamlit UI.

문의 데이터 검증 및 저장은 src.inquiry가 담당하고,
이 모듈은 문의 폼과 화면 상태만 관리한다.
"""

from __future__ import annotations

import logging

import streamlit as st

from src.inquiry import (
    INQUIRY_CATEGORIES,
    InquiryError,
    create_inquiry_record,
    save_user_inquiry,
)


logger = logging.getLogger(__name__)


def render_inquiry_form(
    current_page: str,
) -> None:
    """문의 입력 폼을 표시하고 검증된 문의를 저장한다."""

    with st.form(
        "user_inquiry_form",
        clear_on_submit=True,
        border=True,
    ):
        category = st.selectbox(
            "문의 유형",
            options=list(
                INQUIRY_CATEGORIES
            ),
        )

        message = st.text_area(
            "문의 내용",
            placeholder=(
                "궁금한 점, 오류 상황, 불편한 점 또는 "
                "개선 의견을 적어주세요."
            ),
            height=160,
            max_chars=5000,
        )

        email = st.text_input(
            "답변 받을 이메일 (선택)",
            placeholder="name@example.com",
        )

        st.caption(
            "이메일을 입력하면 문의 기록에 함께 저장됩니다. "
            "답변이 필요한 경우에만 입력해 주세요."
        )

        submitted = st.form_submit_button(
            "문의 보내기",
            type="primary",
            width="stretch",
        )

    if not submitted:
        return

    try:
        record = create_inquiry_record(
            category=category,
            message=message,
            current_page=current_page,
            email=email,
        )

        save_user_inquiry(
            record
        )

    except InquiryError as exc:
        st.warning(
            str(exc)
        )
        return

    except Exception:
        logger.exception(
            "사용자 문의 저장 실패"
        )

        st.error(
            "문의를 저장하는 중 문제가 발생했습니다. "
            "잠시 후 다시 시도해 주세요."
        )

        st.caption(
            "같은 문제가 반복되면 잠시 후 다시 이용해 주세요."
        )
        return

    st.session_state[
        "inquiry_success_id"
    ] = record["inquiry_id"]

    st.session_state[
        "inquiry_open"
    ] = False

    st.rerun()


def render_inquiry_section(
    current_page: str,
) -> None:
    """서비스 페이지 하단의 공통 문의 영역을 표시한다."""

    st.divider()

    success_id = (
        st.session_state.pop(
            "inquiry_success_id",
            None,
        )
    )

    if success_id:
        st.success(
            "문의가 접수되었습니다. "
            f"문의 번호: {success_id}"
        )

    text_col, button_col = (
        st.columns(
            [3, 1],
            vertical_alignment="center",
        )
    )

    with text_col:
        st.subheader(
            "서비스 이용에 도움이 필요하신가요?"
        )

        st.caption(
            "사용 중 궁금한 점이나 불편한 점, "
            "오류 또는 개선 의견을 보내주세요."
        )

    with button_col:
        if st.button(
            "문의하기",
            key="open_inquiry_button",
            width="stretch",
        ):
            st.session_state[
                "inquiry_open"
            ] = (
                not st.session_state.get(
                    "inquiry_open",
                    False,
                )
            )

    if st.session_state.get(
        "inquiry_open",
        False,
    ):
        render_inquiry_form(
            current_page=current_page,
        )