"""src.inquiry의 사용자 문의 계약을 검증한다."""

from __future__ import annotations

import csv

import pytest

from src.config import APP_VERSION
from src.inquiry import (
    INQUIRY_CATEGORIES,
    InquiryError,
    create_inquiry_record,
    save_user_inquiry,
)


def test_create_inquiry_record_contains_required_metadata():
    record = create_inquiry_record(
        category="오류 신고",
        message="예측 결과를 확인하는 중 오류가 발생했습니다.",
        current_page="내 소득 예측",
        email="user@example.com",
        inquiry_id="INQ-TEST0001",
        created_at="2026-08-24T04:00:00Z",
    )

    assert record == {
        "inquiry_id": "INQ-TEST0001",
        "created_at": "2026-08-24T04:00:00Z",
        "version": APP_VERSION,
        "current_page": "내 소득 예측",
        "category": "오류 신고",
        "message": "예측 결과를 확인하는 중 오류가 발생했습니다.",
        "email": "user@example.com",
        "status": "접수",
    }


def test_email_is_optional():
    record = create_inquiry_record(
        category="기타",
        message="문의 내용",
        current_page="소득과의 관계",
    )

    assert record["email"] == ""


def test_invalid_category_is_rejected():
    assert "없는 유형" not in INQUIRY_CATEGORIES

    with pytest.raises(InquiryError):
        create_inquiry_record(
            category="없는 유형",
            message="문의 내용",
            current_page="소득과의 관계",
        )


def test_empty_message_is_rejected():
    with pytest.raises(InquiryError):
        create_inquiry_record(
            category="기타",
            message="   ",
            current_page="소득과의 관계",
        )


def test_invalid_email_is_rejected():
    with pytest.raises(InquiryError):
        create_inquiry_record(
            category="기타",
            message="문의 내용",
            current_page="소득과의 관계",
            email="not-an-email",
        )


def test_save_user_inquiry_creates_and_appends_csv(
    tmp_path,
):
    path = (
        tmp_path
        / "inquiries"
        / "user_inquiries.csv"
    )

    first = create_inquiry_record(
        category="이용 방법 문의",
        message="첫 번째 문의",
        current_page="소득과의 관계",
        inquiry_id="INQ-1",
        created_at="2026-08-24T04:00:00Z",
    )

    second = create_inquiry_record(
        category="기능 개선 요청",
        message="두 번째 문의",
        current_page="내 소득 예측",
        inquiry_id="INQ-2",
        created_at="2026-08-24T04:01:00Z",
    )

    assert save_user_inquiry(
        first,
        path=path,
    ) == path

    save_user_inquiry(
        second,
        path=path,
    )

    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    assert [
        row["inquiry_id"]
        for row in rows
    ] == [
        "INQ-1",
        "INQ-2",
    ]

    assert rows[1]["status"] == "접수"