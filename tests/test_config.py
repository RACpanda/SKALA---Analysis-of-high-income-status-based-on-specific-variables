"""src.config의 v1.5 공통 설정 계약을 검증한다."""

from __future__ import annotations

from src.config import (
    APP_VERSION,
    INQUIRY_DIR,
    OUTPUT_DIR,
)


def test_app_version_is_v1_5():
    assert APP_VERSION == "1.5"


def test_inquiry_directory_is_under_outputs():
    assert (
        INQUIRY_DIR
        == OUTPUT_DIR / "inquiries"
    )