"""Streamlit UI가 내부 예외를 사용자에게 직접 노출하지 않는지 검증한다."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

UI_PATHS = (
    PROJECT_ROOT / "app.py",
    (
        PROJECT_ROOT
        / "src"
        / "ui_association.py"
    ),
    (
        PROJECT_ROOT
        / "src"
        / "ui_prediction.py"
    ),
    (
        PROJECT_ROOT
        / "src"
        / "ui_inquiry.py"
    ),
)


def _read_ui_sources() -> str:
    """오류 처리 정책을 확인할 전체 Streamlit UI 소스를 반환한다."""

    missing = [
        str(path)
        for path in UI_PATHS
        if not path.exists()
    ]

    assert not missing, (
        "검사할 UI 파일이 없습니다: "
        + ", ".join(missing)
    )

    return "\n".join(
        path.read_text(
            encoding="utf-8"
        )
        for path in UI_PATHS
    )


def test_ui_does_not_render_raw_exception_text_for_service_failures():
    source = _read_ui_sources()

    forbidden = [
        "st.error(\n            str(exc)",
        'f"예측 중요도를 표시하지 못했습니다: {exc}"',
        'f"입력값 비교 그래프를 표시하지 못했습니다: {exc}"',
        'f"Adult 데이터를 불러오지 못했습니다: "',
        "st.code(message)",
        "st.exception(",
    ]

    for pattern in forbidden:
        assert pattern not in source


def test_ui_logs_server_side_failures():
    source = _read_ui_sources()

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