"""고소득 예측 모델의 개발·검증용 시각화.

이 모듈은 웹 사용자의 개별 분석 결과를 시각화하지 않는다.

modeling.py가 저장한 테스트셋 평가 산출물을 이용해:
    1. Accuracy / Precision / Recall / F1
    2. ROC curve와 ROC-AUC
    3. Confusion Matrix

를 생성하고 PNG 파일로 저장한다.

사용자용 연관성 분석·예측 시각화는 visualization.py가 담당한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    FIGURE_DIR,
    TABLE_DIR,
    ensure_directories,
)


MODEL_METRICS_PATH = (
    TABLE_DIR
    / "model_metrics.json"
)

MODEL_PREDICTIONS_PATH = (
    TABLE_DIR
    / "model_predictions.csv"
)

INCOME_CLASS_LABELS = [
    "<=50K",
    ">50K",
]


class ModelVisualizationError(
    RuntimeError
):
    """모델 평가 시각화 입력이나 산출물에 문제가 있을 때 발생하는 오류."""


# ============================================================
# 평가 산출물 로딩
# ============================================================

def _require_keys(
    data: dict,
    keys: list[str],
    source_name: str,
) -> None:
    """dict에 필요한 key가 모두 존재하는지 확인한다."""

    missing = [
        key
        for key in keys
        if key not in data
    ]

    if missing:
        raise ModelVisualizationError(
            f"{source_name}에 필요한 항목이 없습니다: "
            f"{missing}"
        )


def _require_columns(
    df: pd.DataFrame,
    columns: list[str],
    source_name: str,
) -> None:
    """DataFrame에 필요한 열이 모두 존재하는지 확인한다."""

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ModelVisualizationError(
            f"{source_name}에 필요한 열이 없습니다: "
            f"{missing}"
        )


def _load_model_artifacts(
) -> tuple[
    dict,
    pd.DataFrame,
]:
    """modeling.py가 저장한 평가 지표와 테스트셋 예측값을 불러온다."""

    if not MODEL_METRICS_PATH.exists():
        raise ModelVisualizationError(
            "모델 평가 지표 파일이 없습니다: "
            f"{MODEL_METRICS_PATH}. "
            "train_income_model()을 먼저 실행하세요."
        )

    if not MODEL_PREDICTIONS_PATH.exists():
        raise ModelVisualizationError(
            "모델 테스트셋 예측 파일이 없습니다: "
            f"{MODEL_PREDICTIONS_PATH}. "
            "train_income_model()을 먼저 실행하세요."
        )

    try:
        metrics = json.loads(
            MODEL_METRICS_PATH.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise ModelVisualizationError(
            "model_metrics.json을 읽을 수 없습니다."
        ) from exc

    if not isinstance(
        metrics,
        dict,
    ):
        raise ModelVisualizationError(
            "model_metrics.json 형식이 올바르지 않습니다."
        )

    try:
        predictions = pd.read_csv(
            MODEL_PREDICTIONS_PATH
        )
    except (
        OSError,
        pd.errors.ParserError,
    ) as exc:
        raise ModelVisualizationError(
            "model_predictions.csv를 읽을 수 없습니다."
        ) from exc

    return (
        metrics,
        predictions,
    )


# ============================================================
# 평가 산출물 검증
# ============================================================

def _prepare_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """테스트셋 예측값의 타입과 허용 범위를 검증한다."""

    _require_columns(
        predictions,
        [
            "y_test",
            "y_pred",
            "y_proba",
        ],
        "model_predictions.csv",
    )

    if predictions.empty:
        raise ModelVisualizationError(
            "model_predictions.csv가 비어 있습니다."
        )

    result = predictions.copy()

    for column in [
        "y_test",
        "y_pred",
        "y_proba",
    ]:
        try:
            result[column] = (
                pd.to_numeric(
                    result[column],
                    errors="raise",
                )
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ModelVisualizationError(
                f"'{column}'을 숫자로 해석할 수 없습니다."
            ) from exc

    y_test_values = set(
        result[
            "y_test"
        ].unique()
    )

    y_pred_values = set(
        result[
            "y_pred"
        ].unique()
    )

    if y_test_values != {
        0,
        1,
    }:
        raise ModelVisualizationError(
            "y_test는 0과 1을 모두 포함해야 합니다. "
            f"현재 값: {y_test_values}"
        )

    if not y_pred_values.issubset(
        {
            0,
            1,
        }
    ):
        raise ModelVisualizationError(
            "y_pred는 0/1 값이어야 합니다. "
            f"현재 값: {y_pred_values}"
        )

    invalid_probability = (
        result[
            "y_proba"
        ].isna()
        | ~np.isfinite(
            result[
                "y_proba"
            ]
        )
        | ~result[
            "y_proba"
        ].between(
            0,
            1,
        )
    )

    if invalid_probability.any():
        raise ModelVisualizationError(
            "y_proba에는 0과 1 사이의 "
            "유한한 확률값만 있어야 합니다."
        )

    result[
        "y_test"
    ] = (
        result[
            "y_test"
        ]
        .astype(int)
    )

    result[
        "y_pred"
    ] = (
        result[
            "y_pred"
        ]
        .astype(int)
    )

    result[
        "y_proba"
    ] = (
        result[
            "y_proba"
        ]
        .astype(float)
    )

    return result


def _calculate_metrics(
    predictions: pd.DataFrame,
) -> dict[str, float]:
    """테스트셋 예측값에서 모델 성능 지표를 다시 계산한다."""

    y_test = (
        predictions[
            "y_test"
        ]
    )

    y_pred = (
        predictions[
            "y_pred"
        ]
    )

    y_proba = (
        predictions[
            "y_proba"
        ]
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_test,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                y_proba,
            )
        ),
    }


def _validate_artifact_consistency(
    metrics: dict,
    predictions: pd.DataFrame,
) -> None:
    """metrics와 predictions가 같은 평가 실행에서 나온 결과인지 확인한다."""

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    _require_keys(
        metrics,
        [
            "test_rows",
            *metric_names,
        ],
        "model_metrics.json",
    )

    try:
        expected_rows = int(
            metrics[
                "test_rows"
            ]
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ModelVisualizationError(
            "model_metrics.json의 test_rows가 올바르지 않습니다."
        ) from exc

    if (
        expected_rows
        != len(
            predictions
        )
    ):
        raise ModelVisualizationError(
            "model_metrics.json과 model_predictions.csv의 "
            "테스트 표본 수가 일치하지 않습니다."
        )

    recalculated = (
        _calculate_metrics(
            predictions
        )
    )

    for metric_name in metric_names:
        try:
            stored_value = float(
                metrics[
                    metric_name
                ]
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ModelVisualizationError(
                f"'{metric_name}' 값이 올바르지 않습니다."
            ) from exc

        if not np.isclose(
            stored_value,
            recalculated[
                metric_name
            ],
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ModelVisualizationError(
                "모델 평가 산출물이 서로 일치하지 않습니다. "
                f"지표: {metric_name}, "
                f"저장값={stored_value:.6f}, "
                f"예측값 재계산={recalculated[metric_name]:.6f}. "
                "train_income_model()을 다시 실행하세요."
            )


# ============================================================
# Figure 저장
# ============================================================

def _save_figure(
    figure: plt.Figure,
    filename: str,
) -> Path:
    """Matplotlib Figure를 개발용 PNG로 저장한다."""

    ensure_directories()

    output_path = (
        FIGURE_DIR
        / filename
    )

    try:
        figure.savefig(
            output_path,
            dpi=160,
            bbox_inches="tight",
        )
    except OSError as exc:
        raise ModelVisualizationError(
            f"모델 진단 그래프를 저장하지 못했습니다: "
            f"{output_path}"
        ) from exc
    finally:
        plt.close(
            figure
        )

    return output_path


# ============================================================
# 성능 지표
# ============================================================

def plot_model_metrics(
    metrics: dict,
) -> Path:
    """Accuracy, Precision, Recall, F1을 막대그래프로 저장한다."""

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    _require_keys(
        metrics,
        metric_names,
        "model_metrics.json",
    )

    values = np.array(
        [
            float(
                metrics[
                    metric
                ]
            )
            for metric
            in metric_names
        ],
        dtype=float,
    )

    if (
        not np.isfinite(
            values
        ).all()
        or (
            values < 0
        ).any()
        or (
            values > 1
        ).any()
    ):
        raise ModelVisualizationError(
            "모델 성능 지표는 0과 1 사이의 "
            "유한한 값이어야 합니다."
        )

    figure, axis = (
        plt.subplots(
            figsize=(
                7,
                5,
            )
        )
    )

    bars = axis.bar(
        metric_names,
        values,
    )

    axis.set_title(
        "Income prediction model performance"
    )

    axis.set_xlabel(
        "Metric"
    )

    axis.set_ylabel(
        "Score"
    )

    axis.set_ylim(
        0,
        1.08,
    )

    for (
        bar,
        value,
    ) in zip(
        bars,
        values,
    ):
        axis.text(
            (
                bar.get_x()
                + bar.get_width()
                / 2
            ),
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    return _save_figure(
        figure,
        "model_performance_metrics.png",
    )


# ============================================================
# ROC Curve
# ============================================================

def plot_roc_curve(
    predictions: pd.DataFrame,
) -> Path:
    """테스트셋 예측확률로 ROC curve와 ROC-AUC를 생성한다."""

    y_test = (
        predictions[
            "y_test"
        ]
    )

    y_proba = (
        predictions[
            "y_proba"
        ]
    )

    false_positive_rate, (
        true_positive_rate
    ), _ = roc_curve(
        y_test,
        y_proba,
    )

    auc = float(
        roc_auc_score(
            y_test,
            y_proba,
        )
    )

    figure, axis = (
        plt.subplots(
            figsize=(
                6,
                6,
            )
        )
    )

    axis.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=(
            f"ROC-AUC = {auc:.3f}"
        ),
    )

    axis.plot(
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
        linestyle="--",
        linewidth=1,
        label="Random classifier",
    )

    axis.set_xlim(
        0,
        1,
    )

    axis.set_ylim(
        0,
        1,
    )

    axis.set_title(
        "ROC curve"
    )

    axis.set_xlabel(
        "False positive rate"
    )

    axis.set_ylabel(
        "True positive rate"
    )

    axis.legend()

    return _save_figure(
        figure,
        "model_roc_curve.png",
    )


# ============================================================
# Confusion Matrix
# ============================================================

def plot_confusion_matrix(
    predictions: pd.DataFrame,
) -> Path:
    """테스트셋의 2×2 Confusion Matrix를 생성한다."""

    confusion = (
        confusion_matrix(
            predictions[
                "y_test"
            ],
            predictions[
                "y_pred"
            ],
            labels=[
                0,
                1,
            ],
        )
    )

    row_totals = (
        confusion
        .sum(
            axis=1,
            keepdims=True,
        )
    )

    percentages = np.divide(
        confusion,
        row_totals,
        out=np.zeros_like(
            confusion,
            dtype=float,
        ),
        where=(
            row_totals
            != 0
        ),
    )

    figure, axis = (
        plt.subplots(
            figsize=(
                6,
                5,
            )
        )
    )

    image = axis.imshow(
        confusion,
        cmap="Blues",
    )

    figure.colorbar(
        image,
        ax=axis,
        label="Count",
    )

    axis.set_xticks(
        [
            0,
            1,
        ],
        labels=(
            INCOME_CLASS_LABELS
        ),
    )

    axis.set_yticks(
        [
            0,
            1,
        ],
        labels=(
            INCOME_CLASS_LABELS
        ),
    )

    axis.set_xlabel(
        "Predicted"
    )

    axis.set_ylabel(
        "Actual"
    )

    axis.set_title(
        "Confusion matrix (test set)"
    )

    threshold = (
        confusion.max()
        / 2
        if confusion.size
        else 0
    )

    for actual in range(
        confusion.shape[
            0
        ]
    ):
        for predicted in range(
            confusion.shape[
                1
            ]
        ):
            count = int(
                confusion[
                    actual,
                    predicted,
                ]
            )

            percentage = float(
                percentages[
                    actual,
                    predicted,
                ]
                * 100
            )

            axis.text(
                predicted,
                actual,
                (
                    f"{count:,}\n"
                    f"({percentage:.1f}%)"
                ),
                ha="center",
                va="center",
                color=(
                    "white"
                    if count
                    > threshold
                    else "black"
                ),
            )

    return _save_figure(
        figure,
        "model_confusion_matrix.png",
    )


# ============================================================
# 전체 모델 진단 시각화
# ============================================================

def create_model_visualizations(
) -> dict[str, Path]:
    """저장된 모델 평가 산출물을 검증하고 개발용 진단 그래프를 생성한다."""

    (
        metrics,
        raw_predictions,
    ) = _load_model_artifacts()

    predictions = (
        _prepare_predictions(
            raw_predictions
        )
    )

    _validate_artifact_consistency(
        metrics,
        predictions,
    )

    return {
        "performance_metrics": (
            plot_model_metrics(
                metrics
            )
        ),
        "roc_curve": (
            plot_roc_curve(
                predictions
            )
        ),
        "confusion_matrix": (
            plot_confusion_matrix(
                predictions
            )
        ),
    }