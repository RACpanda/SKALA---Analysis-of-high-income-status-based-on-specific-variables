"""src.visualization의 v1.5 Plotly 결과 객체 생성 계약을 검증한다."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from src.association import (
    AnalysisRequest,
    analyze_association,
)
from src.visualization import (
    VisualizationError,
    create_association_visualizations,
)


def test_create_association_visualizations_returns_used_figures(
    association_frame,
):
    result = analyze_association(
        association_frame,
        AnalysisRequest(
            exposure="sex",
            controls=("age",),
        ),
    )

    figures = (
        create_association_visualizations(
            result
        )
    )

    assert set(
        figures
    ) == {
        "adjusted_probability",
    }

    assert all(
        isinstance(
            figure,
            go.Figure,
        )
        for figure in figures.values()
    )


def test_psm_result_adds_balance_figure(
    association_frame,
):
    result = analyze_association(
        association_frame,
        AnalysisRequest(
            exposure="sex",
            controls=(
                "age",
                "race",
            ),
            include_psm=True,
        ),
    )

    figures = (
        create_association_visualizations(
            result
        )
    )

    assert (
        "adjusted_probability"
        in figures
    )

    assert (
        "psm_balance"
        in figures
    )

    assert isinstance(
        figures[
            "psm_balance"
        ],
        go.Figure,
    )


def test_invalid_association_result_raises_visualization_error():
    with pytest.raises(
        VisualizationError
    ):
        create_association_visualizations(
            {}
        )