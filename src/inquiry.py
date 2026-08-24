"""사용자 문의 생성·검증·저장 기능.

UI와 저장 로직을 분리하기 위해 Streamlit 코드는 포함하지 않는다.
현재 기본 저장소는 로컬 CSV이며 개발/로컬 검증용이다.
웹 배포 시에는 ``save_user_inquiry`` 구현을 DB/API 등 영속 저장소로
교체할 수 있도록 문의 레코드 계약을 별도 함수로 유지한다.
"""

from __future__ import annotations

import csv
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from src.config import OUTPUT_DIR


SERVICE_VERSION = "1.4"

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

DEFAULT_INQUIRY_PATH = (
    OUTPUT_DIR
    / "inquiries"
    / "user_inquiries.csv"
)

_EMAIL_PATTERN = re.compile(
    r"^[^\s@]+@[^\s@]+\.[^\s@]+$"
)


class InquiryError(ValueError):
    """문의 입력 검증 또는 저장 계약 오류."""


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

    normalized_email = _normalize_email(
        email
    )

    record_id = (
        str(inquiry_id).strip()
        if inquiry_id
        else (
            "INQ-"
            + uuid.uuid4().hex[:12].upper()
        )
    )

    timestamp = (
        str(created_at).strip()
        if created_at
        else (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
    )

    return {
        "inquiry_id": record_id,
        "created_at": timestamp,
        "version": SERVICE_VERSION,
        "current_page": normalized_page,
        "category": normalized_category,
        "message": normalized_message,
        "email": normalized_email,
        "status": "접수",
    }


def save_user_inquiry(
    record: Mapping[str, object],
    *,
    path: Path | str | None = None,
) -> Path:
    """문의 레코드를 로컬 CSV에 저장한다.

    이 함수는 개발/로컬 실행을 위한 기본 저장 어댑터다.
    영속 파일시스템이 보장되지 않는 웹 배포 환경에서는 DB/API 기반
    구현으로 교체해야 한다.
    """

    missing_fields = [
        field
        for field in INQUIRY_FIELDS
        if field not in record
    ]

    if missing_fields:
        raise InquiryError(
            "문의 저장에 필요한 항목이 없습니다: "
            + ", ".join(missing_fields)
        )

    destination = Path(
        path
        if path is not None
        else DEFAULT_INQUIRY_PATH
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    is_new_file = (
        not destination.exists()
        or destination.stat().st_size == 0
    )

    mode = "w" if is_new_file else "a"
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
