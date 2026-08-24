"""Streamlit UI가 기술 예외 내용을 그대로 사용자에게 노출하지 않는지 검증한다."""

from __future__ import annotations

from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_does_not_render_raw_exception_text_for_service_failures():
    source = APP_PATH.read_text(encoding="utf-8")

    forbidden = [
        "st.error(\n            str(exc)",
        'f"예측 중요도를 표시하지 못했습니다: {exc}"',
        'f"입력값 비교 그래프를 표시하지 못했습니다: {exc}"',
        'f"Adult 데이터를 불러오지 못했습니다: "',
        "st.code(message)",
    ]

    for pattern in forbidden:
        assert pattern not in source


def test_app_logs_server_side_failures():
    source = APP_PATH.read_text(encoding="utf-8")

    expected_log_messages = [
        "예측 입력 스키마 로드 실패",
        "개인 소득 예측 실행 실패",
        "개인 예측 설명 그래프 생성 실패",
        "전체 모델 중요도 표시 실패",
        "What-if 시뮬레이션 실행 실패",
        "사용자 문의 저장 실패",
        "연관성 분석 실행 실패",
        "서비스 데이터 로드 실패",
    ]

    for message in expected_log_messages:
        assert message in source
