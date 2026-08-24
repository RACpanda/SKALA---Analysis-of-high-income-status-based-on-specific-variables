"""사용자 문의 생성·검증·저장 기능.

로컬 개발에서는 CSV를 사용할 수 있고,
배포 환경에서는 Supabase Data API를 영속 저장소로 사용한다.

저장 방식:
- csv
- supabase
"""

from __future__ import annotations

import csv
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import httpx

from src.config import (
    APP_VERSION,
    INQUIRY_DIR,
)


logger = logging.getLogger(__name__)


INQUIRY_CATEGORIES = (
    "이용 방법 문의",
    "분석 결과 문의",
    "예측 결과 문의",
    "오류 신고",
    "불편 사항",
    "기능 개선 요청",
    "기타",
)

INQUIRY_FIELDS = (
    "inquiry_id",
    "created_at",
    "version",
    "current_page",
    "category",
    "message",
    "email",
    "status",
)

INQUIRY_STORAGE_CSV = "csv"
INQUIRY_STORAGE_SUPABASE = "supabase"
INQUIRY_TABLE_NAME = "user_inquiries"

DEFAULT_INQUIRY_PATH = (
    INQUIRY_DIR
    / "user_inquiries.csv"
)

SUPABASE_REQUEST_TIMEOUT_SECONDS = 10.0

_EMAIL_PATTERN = re.compile(
    r"^[^\s@]+@[^\s@]+\.[^\s@]+$"
)


class InquiryError(ValueError):
    """문의 입력 검증 또는 저장 오류."""


def _get_secret(
    name: str,
) -> str:
    """환경변수 또는 Streamlit secrets에서 설정값을 읽는다."""

    value = os.getenv(name)

    if value:
        return str(value).strip()

    try:
        import streamlit as st

        if name in st.secrets:
            return str(
                st.secrets[name]
            ).strip()

    except Exception:
        # pytest 또는 Streamlit 외부 실행 환경에서는
        # secrets가 없을 수 있으므로 정상적으로 넘어간다.
        pass

    return ""


def _normalize_email(
    email: str | None,
) -> str:
    """선택 입력 이메일을 정리하고 기본 형식을 검증한다."""

    if email is None:
        return ""

    normalized = str(email).strip()

    if not normalized:
        return ""

    if len(normalized) > 254:
        raise InquiryError(
            "이메일 주소가 너무 깁니다."
        )

    if not _EMAIL_PATTERN.fullmatch(
        normalized
    ):
        raise InquiryError(
            "이메일 주소 형식을 확인해 주세요."
        )

    return normalized


def create_inquiry_record(
    *,
    category: str,
    message: str,
    current_page: str,
    email: str | None = None,
    inquiry_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, str]:
    """사용자 입력을 검증하고 저장 가능한 문의 레코드를 만든다."""

    normalized_category = str(
        category
    ).strip()

    if (
        normalized_category
        not in INQUIRY_CATEGORIES
    ):
        raise InquiryError(
            "문의 유형을 다시 선택해 주세요."
        )

    normalized_message = str(
        message
    ).strip()

    if not normalized_message:
        raise InquiryError(
            "문의 내용을 입력해 주세요."
        )

    if len(normalized_message) > 5000:
        raise InquiryError(
            "문의 내용은 5,000자 이내로 입력해 주세요."
        )

    normalized_page = str(
        current_page
    ).strip()

    if not normalized_page:
        normalized_page = "알 수 없음"

    normalized_email = (
        _normalize_email(email)
    )

    record_id = (
        str(inquiry_id).strip()
        if inquiry_id
        else (
            "INQ-"
            + uuid.uuid4()
            .hex[:12]
            .upper()
        )
    )

    timestamp = (
        str(created_at).strip()
        if created_at
        else (
            datetime.now(
                timezone.utc
            )
            .isoformat(
                timespec="seconds"
            )
            .replace(
                "+00:00",
                "Z",
            )
        )
    )

    return {
        "inquiry_id": record_id,
        "created_at": timestamp,
        "version": APP_VERSION,
        "current_page": normalized_page,
        "category": normalized_category,
        "message": normalized_message,
        "email": normalized_email,
        "status": "접수",
    }


def _validate_record(
    record: Mapping[str, object],
) -> None:
    """저장 전에 필수 필드 존재 여부를 확인한다."""

    missing_fields = [
        field
        for field in INQUIRY_FIELDS
        if field not in record
    ]

    if missing_fields:
        raise InquiryError(
            "문의 저장에 필요한 항목이 없습니다: "
            + ", ".join(
                missing_fields
            )
        )


def _save_to_csv(
    record: Mapping[str, object],
    destination: Path,
) -> Path:
    """문의 레코드를 로컬 CSV에 저장한다."""

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    is_new_file = (
        not destination.exists()
        or destination.stat().st_size == 0
    )

    mode = (
        "w"
        if is_new_file
        else "a"
    )

    encoding = (
        "utf-8-sig"
        if is_new_file
        else "utf-8"
    )

    with destination.open(
        mode,
        newline="",
        encoding=encoding,
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                INQUIRY_FIELDS
            ),
            extrasaction="ignore",
        )

        if is_new_file:
            writer.writeheader()

        writer.writerow(
            {
                field: record[field]
                for field in INQUIRY_FIELDS
            }
        )

    return destination


def _save_to_supabase(
    record: Mapping[str, object],
) -> str:
    """Supabase Data API를 통해 문의를 저장한다."""

    supabase_url = (
        _get_secret(
            "SUPABASE_URL"
        )
        .rstrip("/")
    )

    supabase_key = _get_secret(
        "SUPABASE_KEY"
    )

    if not supabase_url:
        raise InquiryError(
            "Supabase URL이 설정되지 않았습니다."
        )

    if not supabase_key:
        raise InquiryError(
            "Supabase API Key가 설정되지 않았습니다."
        )

    payload = {
        field: record[field]
        for field in INQUIRY_FIELDS
    }

    # 이메일 미입력은 빈 문자열 대신 DB NULL로 저장한다.
    if not str(
        payload.get(
            "email",
            "",
        )
    ).strip():
        payload["email"] = None

    endpoint = (
        f"{supabase_url}"
        f"/rest/v1/{INQUIRY_TABLE_NAME}"
    )

    headers = {
        "apikey": supabase_key,
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    try:
        response = httpx.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=(
                SUPABASE_REQUEST_TIMEOUT_SECONDS
            ),
        )

        response.raise_for_status()

    except httpx.HTTPStatusError as exc:
        # Publishable key 자체는 로그에 출력하지 않는다.
        logger.error(
            "Supabase 문의 저장 HTTP 오류 "
            "status=%s body=%s",
            exc.response.status_code,
            exc.response.text[:1000],
        )

        raise InquiryError(
            "문의 저장소에서 요청을 처리하지 못했습니다."
        ) from exc

    except httpx.RequestError as exc:
        logger.exception(
            "Supabase 문의 저장 네트워크 오류"
        )

        raise InquiryError(
            "문의 저장소에 연결하지 못했습니다."
        ) from exc

    except Exception as exc:
        logger.exception(
            "Supabase 문의 저장 중 예상하지 못한 오류"
        )

        raise InquiryError(
            "문의 저장 중 오류가 발생했습니다."
        ) from exc

    return (
        f"supabase:{INQUIRY_TABLE_NAME}"
    )


def get_inquiry_storage_mode() -> str:
    """현재 문의 저장 방식을 반환한다."""

    mode = (
        _get_secret(
            "INQUIRY_STORAGE_MODE"
        )
        or INQUIRY_STORAGE_CSV
    )

    return mode.strip().lower()


def save_user_inquiry(
    record: Mapping[str, object],
    *,
    path: Path | str | None = None,
) -> Path | str:
    """문의 레코드를 설정된 저장소에 저장한다.

    path를 직접 지정하면 테스트 및 로컬 검증 목적으로
    저장 모드와 관계없이 CSV를 사용한다.
    """

    _validate_record(record)

    if path is not None:
        return _save_to_csv(
            record,
            Path(path),
        )

    storage_mode = (
        get_inquiry_storage_mode()
    )

    if storage_mode == INQUIRY_STORAGE_CSV:
        return _save_to_csv(
            record,
            DEFAULT_INQUIRY_PATH,
        )

    if (
        storage_mode
        == INQUIRY_STORAGE_SUPABASE
    ):
        return _save_to_supabase(
            record
        )

    raise InquiryError(
        "지원하지 않는 문의 저장 방식입니다: "
        f"{storage_mode}"
    )