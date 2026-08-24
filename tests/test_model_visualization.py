"""src.model_visualization의 v1.4 산출물 검증·PNG 생성 계약을 검증한다."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import src.model_visualization as model_visualization
from src.model_visualization import (
    ModelVisualizationError,
    create_model_visualizations,
)


def _write_consistent_model_outputs() -> None:
    y_test = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    y_proba = np.array([0.05, 0.20, 0.40, 0.70, 0.35, 0.60, 0.80, 0.95])
    y_pred = (y_proba >= 0.50).astype(int)

    predictions = pd.DataFrame(
        {
            "row_id": np.arange(len(y_test)),
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
    )

    metrics = {
        "test_rows": len(predictions),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    model_visualization.MODEL_METRICS_PATH.write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )
    predictions.to_csv(
        model_visualization.MODEL_PREDICTIONS_PATH,
        index=False,
    )


def test_create_model_visualizations_writes_expected_figures():
    _write_consistent_model_outputs()

    outputs = create_model_visualizations()

    assert set(outputs) == {
        "performance_metrics",
        "roc_curve",
        "confusion_matrix",
    }

    expected_names = {
        "model_performance_metrics.png",
        "model_roc_curve.png",
        "model_confusion_matrix.png",
    }
    assert {path.name for path in outputs.values()} == expected_names
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs.values())


def test_create_model_visualizations_rejects_missing_model_outputs():
    with pytest.raises(ModelVisualizationError, match="모델 평가 지표 파일"):
        create_model_visualizations()


def test_create_model_visualizations_rejects_inconsistent_metrics():
    _write_consistent_model_outputs()

    metrics = json.loads(
        model_visualization.MODEL_METRICS_PATH.read_text(encoding="utf-8")
    )
    metrics["accuracy"] = 0.0
    model_visualization.MODEL_METRICS_PATH.write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )

    with pytest.raises(ModelVisualizationError, match="서로 일치하지 않습니다"):
        create_model_visualizations()
