
"""v1.5 Streamlit UI 모듈 분리 구조를 검증한다."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

APP_PATH = (
    PROJECT_ROOT
    / "app.py"
)

ASSOCIATION_UI_PATH = (
    PROJECT_ROOT
    / "src"
    / "ui_association.py"
)

PREDICTION_UI_PATH = (
    PROJECT_ROOT
    / "src"
    / "ui_prediction.py"
)


def test_app_delegates_page_ui_to_modules():
    source = APP_PATH.read_text(
        encoding="utf-8"
    )

    # 앱 진입점에 공통 데이터 로더가 반드시 남아 있어야 한다.
    assert (
        "def load_service_data("
        in source
    )

    assert (
        "from src.data import load_and_clean"
        in source
    )

    assert (
        "from src.ui_association "
        "import association_page"
        in source
    )

    assert (
        "from src.ui_prediction "
        "import prediction_page"
        in source
    )

    forbidden_definitions = [
        "def display_unadjusted_result(",
        "def display_adjusted_result(",
        "def display_psm_result(",
        "def association_page(",
        "def display_association_error(",
        "def _continuous_input_widget(",
        "def _categorical_input_widget(",
        "def prediction_page(",
    ]

    for definition in forbidden_definitions:
        assert definition not in source

    assert (
        "from src.ui_association "
        "import association_page"
        in source
    )

    assert (
        "from src.ui_prediction "
        "import prediction_page"
        in source
    )

    forbidden_definitions = [
        "def display_unadjusted_result(",
        "def display_adjusted_result(",
        "def display_psm_result(",
        "def association_page(",
        "def display_association_error(",
        "def _continuous_input_widget(",
        "def _categorical_input_widget(",
        "def prediction_page(",
    ]

    for definition in forbidden_definitions:
        assert definition not in source


def test_page_modules_own_their_ui_functions():
    association_source = (
        ASSOCIATION_UI_PATH.read_text(
            encoding="utf-8"
        )
    )

    prediction_source = (
        PREDICTION_UI_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        "def association_page("
        in association_source
    )

    assert (
        "def display_association_error("
        in association_source
    )

    assert (
        "def prediction_page("
        in prediction_source
    )

    assert (
        "def load_prediction_schema("
        in prediction_source
    )