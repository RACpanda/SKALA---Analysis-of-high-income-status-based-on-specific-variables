"""src.inquiry의 사용자 문의 생성·저장 계약을 검증한다.

Supabase 관련 테스트는 실제 외부 서버에 접근하지 않는다.
httpx 요청을 monkeypatch하여 성공·인증 실패·권한 실패·서버 오류·
네트워크 오류를 재현한다.
"""

from __future__ import annotations

import csv
import logging

import httpx
import pytest

import src.inquiry as inquiry

from src.config import APP_VERSION
from src.inquiry import (
    INQUIRY_CATEGORIES,
    INQUIRY_STORAGE_CSV,
    INQUIRY_STORAGE_SUPABASE,
    INQUIRY_TABLE_NAME,
    SUPABASE_REQUEST_TIMEOUT_SECONDS,
    InquiryError,
    create_inquiry_record,
    get_inquiry_storage_mode,
    save_user_inquiry,
)


# ============================================================
# 공통 테스트 데이터
# ============================================================

TEST_SUPABASE_URL = (
    "https://example.supabase.co"
)

TEST_SUPABASE_KEY = (
    "sb_publishable_test_key"
)


def _record(
    *,
    email: str | None = "user@example.com",
) -> dict[str, str]:
    """테스트용 문의 레코드를 생성한다."""

    return create_inquiry_record(
        category="오류 신고",
        message=(
            "예측 결과를 확인하는 중 "
            "오류가 발생했습니다."
        ),
        current_page="내 소득 예측",
        email=email,
        inquiry_id="INQ-TEST0001",
        created_at="2026-08-24T04:00:00Z",
    )


def _configure_supabase(
    monkeypatch,
    *,
    url: str = TEST_SUPABASE_URL,
    key: str = TEST_SUPABASE_KEY,
) -> None:
    """Supabase 관련 secret 값을 테스트용으로 대체한다."""

    def fake_get_secret(
        name: str,
    ) -> str:
        values = {
            "INQUIRY_STORAGE_MODE": (
                INQUIRY_STORAGE_SUPABASE
            ),
            "SUPABASE_URL": url,
            "SUPABASE_KEY": key,
        }

        return values.get(
            name,
            "",
        )

    monkeypatch.setattr(
        inquiry,
        "_get_secret",
        fake_get_secret,
    )


# ============================================================
# 문의 레코드 생성
# ============================================================

def test_create_inquiry_record_contains_required_metadata():
    record = _record()

    assert record == {
        "inquiry_id": "INQ-TEST0001",
        "created_at": (
            "2026-08-24T04:00:00Z"
        ),
        "version": APP_VERSION,
        "current_page": "내 소득 예측",
        "category": "오류 신고",
        "message": (
            "예측 결과를 확인하는 중 "
            "오류가 발생했습니다."
        ),
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
    assert (
        "없는 유형"
        not in INQUIRY_CATEGORIES
    )

    with pytest.raises(
        InquiryError
    ):
        create_inquiry_record(
            category="없는 유형",
            message="문의 내용",
            current_page="소득과의 관계",
        )


def test_empty_message_is_rejected():
    with pytest.raises(
        InquiryError
    ):
        create_inquiry_record(
            category="기타",
            message="   ",
            current_page="소득과의 관계",
        )


def test_invalid_email_is_rejected():
    with pytest.raises(
        InquiryError
    ):
        create_inquiry_record(
            category="기타",
            message="문의 내용",
            current_page="소득과의 관계",
            email="not-an-email",
        )


# ============================================================
# CSV 저장
# ============================================================

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
            csv.DictReader(
                handle
            )
        )

    assert [
        row["inquiry_id"]
        for row in rows
    ] == [
        "INQ-1",
        "INQ-2",
    ]

    assert (
        rows[1]["status"]
        == "접수"
    )


def test_explicit_path_uses_csv_even_in_supabase_mode(
    tmp_path,
    monkeypatch,
):
    _configure_supabase(
        monkeypatch
    )

    destination = (
        tmp_path
        / "forced.csv"
    )

    result = save_user_inquiry(
        _record(),
        path=destination,
    )

    assert result == destination
    assert destination.exists()


# ============================================================
# 저장 방식 선택
# ============================================================

def test_default_storage_mode_is_csv(
    monkeypatch,
):
    monkeypatch.setattr(
        inquiry,
        "_get_secret",
        lambda name: "",
    )

    assert (
        get_inquiry_storage_mode()
        == INQUIRY_STORAGE_CSV
    )


def test_unknown_storage_mode_is_rejected(
    monkeypatch,
):
    monkeypatch.setattr(
        inquiry,
        "_get_secret",
        lambda name: (
            "unknown"
            if name
            == "INQUIRY_STORAGE_MODE"
            else ""
        ),
    )

    with pytest.raises(
        InquiryError,
        match="지원하지 않는 문의 저장 방식",
    ):
        save_user_inquiry(
            _record()
        )


# ============================================================
# Supabase 설정 검증
# ============================================================

def test_supabase_url_is_required(
    monkeypatch,
):
    _configure_supabase(
        monkeypatch,
        url="",
    )

    with pytest.raises(
        InquiryError,
        match="Supabase URL이 설정되지 않았습니다",
    ):
        save_user_inquiry(
            _record()
        )


def test_supabase_key_is_required(
    monkeypatch,
):
    _configure_supabase(
        monkeypatch,
        key="",
    )

    with pytest.raises(
        InquiryError,
        match=(
            "Supabase API Key가 "
            "설정되지 않았습니다"
        ),
    ):
        save_user_inquiry(
            _record()
        )


# ============================================================
# Supabase 정상 저장
# ============================================================

def test_supabase_insert_uses_expected_data_api_contract(
    monkeypatch,
):
    _configure_supabase(
        monkeypatch
    )

    captured = {}

    def fake_post(
        url,
        *,
        headers,
        json,
        timeout,
    ):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout

        request = httpx.Request(
            "POST",
            url,
        )

        return httpx.Response(
            status_code=201,
            request=request,
        )

    monkeypatch.setattr(
        inquiry.httpx,
        "post",
        fake_post,
    )

    result = save_user_inquiry(
        _record(
            email=None
        )
    )

    assert (
        result
        == (
            f"supabase:"
            f"{INQUIRY_TABLE_NAME}"
        )
    )

    assert captured["url"] == (
        f"{TEST_SUPABASE_URL}"
        f"/rest/v1/"
        f"{INQUIRY_TABLE_NAME}"
    )

    assert captured["headers"] == {
        "apikey": TEST_SUPABASE_KEY,
        "Content-Type": (
            "application/json"
        ),
        "Prefer": "return=minimal",
    }

    assert (
        captured["timeout"]
        == SUPABASE_REQUEST_TIMEOUT_SECONDS
    )

    payload = captured["json"]

    assert (
        payload["inquiry_id"]
        == "INQ-TEST0001"
    )

    assert (
        payload["version"]
        == APP_VERSION
    )

    assert (
        payload["category"]
        == "오류 신고"
    )

    assert (
        payload["status"]
        == "접수"
    )

    # 이메일 미입력 시
    # Supabase에는 빈 문자열이 아닌 NULL을 저장한다.
    assert payload["email"] is None


# ============================================================
# Supabase HTTP 오류
# ============================================================

@pytest.mark.parametrize(
    "status_code",
    [
        401,
        403,
        500,
    ],
)
def test_supabase_http_error_is_converted_to_safe_user_error(
    monkeypatch,
    caplog,
    status_code,
):
    _configure_supabase(
        monkeypatch
    )

    def fake_post(
        url,
        *,
        headers,
        json,
        timeout,
    ):
        request = httpx.Request(
            "POST",
            url,
        )

        return httpx.Response(
            status_code=status_code,
            request=request,
            text=(
                '{"message":"request failed"}'
            ),
        )

    monkeypatch.setattr(
        inquiry.httpx,
        "post",
        fake_post,
    )

    with caplog.at_level(
        logging.ERROR
    ):
        with pytest.raises(
            InquiryError,
            match=(
                "문의 저장소에서 요청을 "
                "처리하지 못했습니다"
            ),
        ):
            save_user_inquiry(
                _record()
            )

    # 사용자에게 보여줄 예외나 개발 로그에
    # Publishable key 자체를 기록하지 않는다.
    assert (
        TEST_SUPABASE_KEY
        not in caplog.text
    )

    assert (
        f"status={status_code}"
        in caplog.text
    )


# ============================================================
# Supabase 네트워크 오류
# ============================================================

def test_supabase_network_error_is_converted_to_safe_user_error(
    monkeypatch,
):
    _configure_supabase(
        monkeypatch
    )

    def fake_post(
        url,
        *,
        headers,
        json,
        timeout,
    ):
        request = httpx.Request(
            "POST",
            url,
        )

        raise httpx.ConnectError(
            "connection failed",
            request=request,
        )

    monkeypatch.setattr(
        inquiry.httpx,
        "post",
        fake_post,
    )

    with pytest.raises(
        InquiryError,
        match=(
            "문의 저장소에 "
            "연결하지 못했습니다"
        ),
    ):
        save_user_inquiry(
            _record()
        )