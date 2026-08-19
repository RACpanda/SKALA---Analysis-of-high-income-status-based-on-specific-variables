"""Version 1.2 probability calibration 실험.

현재 production HistGradientBoosting 모델의 확률 신뢰도를
학습 데이터 내부 Out-of-Fold 예측으로 평가한다.

비교 대상:
    1. Uncalibrated HistGradientBoosting
    2. Sigmoid Calibration
    3. Isotonic Calibration

중요:
    - held-out test set은 사용하지 않는다.
    - 모델 파라미터는 Step 1 결과에 따라 기존 MODEL_PARAMS를 유지한다.
    - 최종 배포 모델은 아직 변경하지 않는다.
"""

from __future__ import annotations

from pathlib import Path
import sys


# ============================================================
# 프로젝트 루트 import 경로
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import json
import time

import numpy as np
import pandas as pd

from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve,
)
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from src.config import (
    PREDICTION_FEATURE_COLUMNS,
    RANDOM_STATE,
    TABLE_DIR,
    TARGET_COLUMN,
    ensure_directories,
)
from src.data import load_and_clean
from src.modeling import (
    MODEL_PARAMS,
    _add_training_references,
    _build_input_schema,
    _build_pipeline,
    _coerce_features,
    _feature_type_columns,
    _prepare_training_data,
)


# ============================================================
# 실험 설정
# ============================================================

OUTER_CV_FOLDS = 5
CALIBRATION_CV_FOLDS = 3
CALIBRATION_BINS = 10


# ============================================================
# 평가 함수
# ============================================================

def _evaluate_probabilities(
    y_true: pd.Series,
    probability: np.ndarray,
) -> dict:
    """확률 예측의 discrimination과 calibration 지표를 계산한다."""

    probability = np.asarray(
        probability,
        dtype=float,
    )

    if (
        probability.ndim != 1
        or len(probability) != len(y_true)
    ):
        raise ValueError(
            "확률 예측 배열의 형태가 올바르지 않습니다."
        )

    if not np.isfinite(
        probability
    ).all():
        raise ValueError(
            "확률 예측에 유한하지 않은 값이 있습니다."
        )

    if (
        (probability < 0)
        | (probability > 1)
    ).any():
        raise ValueError(
            "확률 예측이 0~1 범위를 벗어났습니다."
        )

    actual_rate = float(
        np.mean(
            y_true
        )
    )

    mean_probability = float(
        np.mean(
            probability
        )
    )

    return {
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probability,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                probability,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probability,
                labels=[
                    0,
                    1,
                ],
            )
        ),
        "actual_positive_rate": (
            actual_rate
        ),
        "mean_predicted_probability": (
            mean_probability
        ),
        "mean_probability_bias": float(
            mean_probability
            - actual_rate
        ),
    }


def _calibration_records(
    y_true: pd.Series,
    probability: np.ndarray,
    model_name: str,
) -> list[dict]:
    """Calibration curve의 각 bin 결과를 저장 가능한 형태로 변환한다."""

    fraction_positive, mean_predicted = (
        calibration_curve(
            y_true,
            probability,
            n_bins=CALIBRATION_BINS,
            strategy="quantile",
        )
    )

    records: list[dict] = []

    for (
        index,
        (
            predicted,
            observed,
        ),
    ) in enumerate(
        zip(
            mean_predicted,
            fraction_positive,
        ),
        start=1,
    ):
        records.append(
            {
                "model": (
                    model_name
                ),
                "bin": int(
                    index
                ),
                "mean_predicted_probability": float(
                    predicted
                ),
                "observed_positive_rate": float(
                    observed
                ),
                "calibration_gap": float(
                    predicted
                    - observed
                ),
            }
        )

    return records


def _print_metrics(
    name: str,
    metrics: dict,
) -> None:
    """모델별 calibration 평가 결과를 출력한다."""

    print(
        f"{name}"
    )

    print(
        f"  ROC-AUC               : "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"  Brier Score           : "
        f"{metrics['brier_score']:.6f}"
    )

    print(
        f"  Log Loss              : "
        f"{metrics['log_loss']:.6f}"
    )

    print(
        f"  Actual positive rate  : "
        f"{metrics['actual_positive_rate']:.6f}"
    )

    print(
        f"  Mean predicted prob.  : "
        f"{metrics['mean_predicted_probability']:.6f}"
    )

    print(
        f"  Mean probability bias : "
        f"{metrics['mean_probability_bias']:+.6f}"
    )


# ============================================================
# Calibrated model 생성
# ============================================================

def _build_calibrated_model(
    *,
    method: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
):
    """현재 production pipeline에 calibration layer를 추가한다."""

    base_pipeline = (
        _build_pipeline(
            numeric_columns,
            categorical_columns,
        )
    )

    inner_cv = (
        StratifiedKFold(
            n_splits=(
                CALIBRATION_CV_FOLDS
            ),
            shuffle=True,
            random_state=(
                RANDOM_STATE
            ),
        )
    )

    return CalibratedClassifierCV(
        estimator=base_pipeline,
        method=method,
        cv=inner_cv,
        ensemble=True,
        n_jobs=-1,
    )


# ============================================================
# 실험
# ============================================================

def main() -> None:
    print(
        "=" * 72
    )

    print(
        "Adult Income Explorer"
    )

    print(
        "Version 1.2 - Probability Calibration"
    )

    print(
        "=" * 72
    )

    # --------------------------------------------------------
    # 현재 production 데이터 계약
    # --------------------------------------------------------

    df = load_and_clean(
        save_output=False,
    )

    model_df = (
        _prepare_training_data(
            df
        )
    )

    raw_X = (
        model_df[
            PREDICTION_FEATURE_COLUMNS
        ]
        .copy()
    )

    y = (
        pd.to_numeric(
            model_df[
                TARGET_COLUMN
            ],
            errors="raise",
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # production과 같은 80:20 split
    #
    # test set은 계속 봉인한다.
    # --------------------------------------------------------

    (
        raw_X_train,
        raw_X_test,
        y_train,
        y_test,
    ) = train_test_split(
        raw_X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print()
    print(
        "[DATA]"
    )

    print(
        f"total rows         : "
        f"{len(raw_X):,}"
    )

    print(
        f"calibration train  : "
        f"{len(raw_X_train):,}"
    )

    print(
        f"reserved test rows : "
        f"{len(raw_X_test):,}"
    )

    print()
    print(
        "held-out test set은 "
        "이번 단계에서도 사용하지 않습니다."
    )

    # --------------------------------------------------------
    # production 입력 schema
    # --------------------------------------------------------

    input_schema = (
        _build_input_schema(
            raw_X_train
        )
    )

    input_schema = (
        _add_training_references(
            input_schema,
            raw_X_train,
        )
    )

    X_train = (
        _coerce_features(
            raw_X_train,
            input_schema,
        )
    )

    (
        numeric_columns,
        categorical_columns,
    ) = (
        _feature_type_columns()
    )

    print()
    print(
        "[MODEL]"
    )

    print(
        "MODEL_PARAMS:"
    )

    print(
        json.dumps(
            MODEL_PARAMS,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Outer CV
    #
    # 모든 비교 모델은 완전히 동일한 split을 사용한다.
    # --------------------------------------------------------

    outer_cv = (
        StratifiedKFold(
            n_splits=(
                OUTER_CV_FOLDS
            ),
            shuffle=True,
            random_state=(
                RANDOM_STATE
            ),
        )
    )

    results: dict[str, dict] = {}
    calibration_records: list[
        dict
    ] = []

    # ========================================================
    # 1. Uncalibrated baseline
    # ========================================================

    print()
    print(
        "[1/3] UNCALIBRATED"
    )

    baseline_model = (
        _build_pipeline(
            numeric_columns,
            categorical_columns,
        )
    )

    start = (
        time.perf_counter()
    )

    baseline_oof = (
        cross_val_predict(
            baseline_model,
            X_train,
            y_train,
            cv=outer_cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    baseline_metrics = (
        _evaluate_probabilities(
            y_train,
            baseline_oof,
        )
    )

    baseline_metrics[
        "elapsed_seconds"
    ] = float(
        elapsed
    )

    results[
        "uncalibrated"
    ] = baseline_metrics

    calibration_records.extend(
        _calibration_records(
            y_train,
            baseline_oof,
            "uncalibrated",
        )
    )

    _print_metrics(
        "Uncalibrated",
        baseline_metrics,
    )

    print(
        f"  Time                  : "
        f"{elapsed:.2f}s"
    )

    # ========================================================
    # 2. Sigmoid
    # ========================================================

    print()
    print(
        "[2/3] SIGMOID CALIBRATION"
    )

    sigmoid_model = (
        _build_calibrated_model(
            method="sigmoid",
            numeric_columns=(
                numeric_columns
            ),
            categorical_columns=(
                categorical_columns
            ),
        )
    )

    start = (
        time.perf_counter()
    )

    sigmoid_oof = (
        cross_val_predict(
            sigmoid_model,
            X_train,
            y_train,
            cv=outer_cv,
            method="predict_proba",
            # 내부 calibration CV가 병렬화되므로
            # 중첩 병렬화를 피하기 위해 outer는 1로 둔다.
            n_jobs=1,
        )[:, 1]
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    sigmoid_metrics = (
        _evaluate_probabilities(
            y_train,
            sigmoid_oof,
        )
    )

    sigmoid_metrics[
        "elapsed_seconds"
    ] = float(
        elapsed
    )

    results[
        "sigmoid"
    ] = sigmoid_metrics

    calibration_records.extend(
        _calibration_records(
            y_train,
            sigmoid_oof,
            "sigmoid",
        )
    )

    _print_metrics(
        "Sigmoid",
        sigmoid_metrics,
    )

    print(
        f"  Time                  : "
        f"{elapsed:.2f}s"
    )

    # ========================================================
    # 3. Isotonic
    # ========================================================

    print()
    print(
        "[3/3] ISOTONIC CALIBRATION"
    )

    isotonic_model = (
        _build_calibrated_model(
            method="isotonic",
            numeric_columns=(
                numeric_columns
            ),
            categorical_columns=(
                categorical_columns
            ),
        )
    )

    start = (
        time.perf_counter()
    )

    isotonic_oof = (
        cross_val_predict(
            isotonic_model,
            X_train,
            y_train,
            cv=outer_cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    isotonic_metrics = (
        _evaluate_probabilities(
            y_train,
            isotonic_oof,
        )
    )

    isotonic_metrics[
        "elapsed_seconds"
    ] = float(
        elapsed
    )

    results[
        "isotonic"
    ] = isotonic_metrics

    calibration_records.extend(
        _calibration_records(
            y_train,
            isotonic_oof,
            "isotonic",
        )
    )

    _print_metrics(
        "Isotonic",
        isotonic_metrics,
    )

    print(
        f"  Time                  : "
        f"{elapsed:.2f}s"
    )

    # ========================================================
    # 비교
    # ========================================================

    print()
    print(
        "[COMPARISON]"
    )

    comparison = (
        pd.DataFrame(
            [
                {
                    "model": name,
                    **metrics,
                }
                for (
                    name,
                    metrics,
                ) in results.items()
            ]
        )
        .sort_values(
            "brier_score"
        )
        .reset_index(
            drop=True
        )
    )

    display_columns = [
        "model",
        "roc_auc",
        "brier_score",
        "log_loss",
        "mean_predicted_probability",
        "mean_probability_bias",
    ]

    print(
        comparison[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    # --------------------------------------------------------
    # baseline 대비 변화
    # --------------------------------------------------------

    baseline_brier = (
        results[
            "uncalibrated"
        ][
            "brier_score"
        ]
    )

    baseline_log_loss = (
        results[
            "uncalibrated"
        ][
            "log_loss"
        ]
    )

    baseline_auc = (
        results[
            "uncalibrated"
        ][
            "roc_auc"
        ]
    )

    print()
    print(
        "[CHANGE VS UNCALIBRATED]"
    )

    for method in [
        "sigmoid",
        "isotonic",
    ]:
        metrics = (
            results[
                method
            ]
        )

        print(
            method
        )

        print(
            "  Brier change : "
            f"{metrics['brier_score'] - baseline_brier:+.6f}"
        )

        print(
            "  LogLoss change: "
            f"{metrics['log_loss'] - baseline_log_loss:+.6f}"
        )

        print(
            "  ROC-AUC change: "
            f"{metrics['roc_auc'] - baseline_auc:+.6f}"
        )

    # ========================================================
    # 저장
    # ========================================================

    ensure_directories()

    summary = {
        "version": "1.2",
        "experiment": (
            "probability_calibration"
        ),
        "outer_cv_folds": (
            OUTER_CV_FOLDS
        ),
        "calibration_cv_folds": (
            CALIBRATION_CV_FOLDS
        ),
        "calibration_bins": (
            CALIBRATION_BINS
        ),
        "evaluation_strategy": (
            "nested_out_of_fold_on_training_data"
        ),
        "train_rows": int(
            len(
                X_train
            )
        ),
        "reserved_test_rows": int(
            len(
                raw_X_test
            )
        ),
        "model_params": dict(
            MODEL_PARAMS
        ),
        "results": results,
    }

    (
        TABLE_DIR
        / "model_calibration_v1_2.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    comparison.to_csv(
        TABLE_DIR
        / "model_calibration_v1_2_comparison.csv",
        index=False,
    )

    pd.DataFrame(
        calibration_records
    ).to_csv(
        TABLE_DIR
        / "model_calibration_v1_2_curve.csv",
        index=False,
    )

    # OOF 확률도 보관
    pd.DataFrame(
        {
            "row_id": (
                X_train.index
            ),
            "y_true": (
                y_train.to_numpy()
            ),
            "uncalibrated_probability": (
                baseline_oof
            ),
            "sigmoid_probability": (
                sigmoid_oof
            ),
            "isotonic_probability": (
                isotonic_oof
            ),
        }
    ).to_csv(
        TABLE_DIR
        / "model_calibration_v1_2_oof_predictions.csv",
        index=False,
    )

    print()
    print(
        "[SAVED]"
    )

    for filename in [
        "model_calibration_v1_2.json",
        "model_calibration_v1_2_comparison.csv",
        "model_calibration_v1_2_curve.csv",
        "model_calibration_v1_2_oof_predictions.csv",
    ]:
        print(
            TABLE_DIR
            / filename
        )

    print()
    print(
        "=" * 72
    )

    print(
        "STEP 2 EXPERIMENT COMPLETE"
    )

    print(
        "아직 production 모델은 "
        "변경하지 않았습니다."
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()