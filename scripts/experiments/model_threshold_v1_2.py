"""Version 1.2 classification threshold 실험.

Step 2에서 생성한 Sigmoid calibrated OOF probability를 사용하여
classification threshold에 따른 성능 변화를 비교한다.

중요:
    - held-out test set을 사용하지 않는다.
    - threshold 선택은 training OOF prediction만 이용한다.
    - production 모델은 아직 변경하지 않는다.
"""

from __future__ import annotations

from pathlib import Path
import sys


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

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    TABLE_DIR,
    ensure_directories,
)


# ============================================================
# 설정
# ============================================================

OOF_PATH = (
    TABLE_DIR
    / "model_calibration_v1_2_oof_predictions.csv"
)

THRESHOLDS = np.round(
    np.arange(
        0.05,
        0.951,
        0.01,
    ),
    2,
)


# ============================================================
# Threshold 평가
# ============================================================

def _evaluate_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict:
    prediction = (
        probability
        >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            prediction,
            labels=[
                0,
                1,
            ],
        )
        .ravel()
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "threshold": float(
            threshold
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                prediction,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "specificity": float(
            specificity
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                prediction,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "tn": int(
            tn
        ),
        "fp": int(
            fp
        ),
        "fn": int(
            fn
        ),
        "tp": int(
            tp
        ),
        "predicted_positive_rate": float(
            prediction.mean()
        ),
    }


def _print_result(
    title: str,
    row: pd.Series,
) -> None:
    print()
    print(
        title
    )

    print(
        f"  threshold         : "
        f"{row['threshold']:.2f}"
    )

    print(
        f"  accuracy          : "
        f"{row['accuracy']:.6f}"
    )

    print(
        f"  precision         : "
        f"{row['precision']:.6f}"
    )

    print(
        f"  recall            : "
        f"{row['recall']:.6f}"
    )

    print(
        f"  specificity       : "
        f"{row['specificity']:.6f}"
    )

    print(
        f"  balanced accuracy : "
        f"{row['balanced_accuracy']:.6f}"
    )

    print(
        f"  F1                : "
        f"{row['f1']:.6f}"
    )

    print(
        f"  predicted positive: "
        f"{row['predicted_positive_rate']:.6f}"
    )

    print(
        "  confusion matrix  : "
        f"TN={int(row['tn'])}, "
        f"FP={int(row['fp'])}, "
        f"FN={int(row['fn'])}, "
        f"TP={int(row['tp'])}"
    )


# ============================================================
# 실행
# ============================================================

def main() -> None:
    print(
        "=" * 72
    )

    print(
        "Adult Income Explorer"
    )

    print(
        "Version 1.2 - Classification Threshold"
    )

    print(
        "=" * 72
    )

    if not OOF_PATH.exists():
        raise FileNotFoundError(
            "Calibration OOF 결과가 없습니다: "
            f"{OOF_PATH}"
        )

    df = pd.read_csv(
        OOF_PATH
    )

    required_columns = {
        "y_true",
        "sigmoid_probability",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Threshold 실험에 필요한 컬럼이 없습니다: "
            f"{sorted(missing)}"
        )

    y_true = (
        pd.to_numeric(
            df[
                "y_true"
            ],
            errors="raise",
        )
        .astype(int)
        .to_numpy()
    )

    probability = (
        pd.to_numeric(
            df[
                "sigmoid_probability"
            ],
            errors="raise",
        )
        .astype(float)
        .to_numpy()
    )

    if not np.isfinite(
        probability
    ).all():
        raise ValueError(
            "Sigmoid probability에 "
            "유한하지 않은 값이 있습니다."
        )

    print()
    print(
        "[DATA]"
    )

    print(
        f"OOF rows             : "
        f"{len(df):,}"
    )

    print(
        f"Actual positive rate : "
        f"{y_true.mean():.6f}"
    )

    # --------------------------------------------------------
    # 전체 threshold 탐색
    # --------------------------------------------------------

    rows = [
        _evaluate_threshold(
            y_true,
            probability,
            threshold,
        )
        for threshold
        in THRESHOLDS
    ]

    results = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # 0.5 기준
    # --------------------------------------------------------

    default_row = (
        results.loc[
            np.isclose(
                results[
                    "threshold"
                ],
                0.50,
            )
        ]
        .iloc[0]
    )

    # --------------------------------------------------------
    # F1 최대
    # --------------------------------------------------------

    best_f1_row = (
        results.loc[
            results[
                "f1"
            ].idxmax()
        ]
    )

    # --------------------------------------------------------
    # Balanced Accuracy 최대
    # --------------------------------------------------------

    best_balanced_row = (
        results.loc[
            results[
                "balanced_accuracy"
            ].idxmax()
        ]
    )

    # --------------------------------------------------------
    # Precision / Recall 차이가 가장 작은 threshold
    # --------------------------------------------------------

    precision_recall_gap = (
        results[
            "precision"
        ]
        .sub(
            results[
                "recall"
            ]
        )
        .abs()
    )

    closest_pr_row = (
        results.loc[
            precision_recall_gap.idxmin()
        ]
    )

    # ========================================================
    # 출력
    # ========================================================

    _print_result(
        "[DEFAULT 0.50]",
        default_row,
    )

    _print_result(
        "[BEST F1]",
        best_f1_row,
    )

    _print_result(
        "[BEST BALANCED ACCURACY]",
        best_balanced_row,
    )

    _print_result(
        "[CLOSEST PRECISION / RECALL]",
        closest_pr_row,
    )

    print()
    print(
        "[CHANGE VS 0.50]"
    )

    for (
        name,
        row,
    ) in [
        (
            "best_f1",
            best_f1_row,
        ),
        (
            "best_balanced_accuracy",
            best_balanced_row,
        ),
        (
            "closest_precision_recall",
            closest_pr_row,
        ),
    ]:
        print()
        print(
            name
        )

        print(
            "  threshold change : "
            f"{row['threshold'] - default_row['threshold']:+.2f}"
        )

        print(
            "  accuracy change  : "
            f"{row['accuracy'] - default_row['accuracy']:+.6f}"
        )

        print(
            "  precision change : "
            f"{row['precision'] - default_row['precision']:+.6f}"
        )

        print(
            "  recall change    : "
            f"{row['recall'] - default_row['recall']:+.6f}"
        )

        print(
            "  F1 change        : "
            f"{row['f1'] - default_row['f1']:+.6f}"
        )

    # ========================================================
    # 저장
    # ========================================================

    ensure_directories()

    results.to_csv(
        TABLE_DIR
        / "model_threshold_v1_2_scan.csv",
        index=False,
    )

    summary = {
        "version": "1.2",
        "experiment": (
            "classification_threshold"
        ),
        "probability_source": (
            "sigmoid_calibrated_oof"
        ),
        "evaluation_rows": int(
            len(df)
        ),
        "default_threshold": (
            default_row.to_dict()
        ),
        "best_f1": (
            best_f1_row.to_dict()
        ),
        "best_balanced_accuracy": (
            best_balanced_row.to_dict()
        ),
        "closest_precision_recall": (
            closest_pr_row.to_dict()
        ),
    }

    (
        TABLE_DIR
        / "model_threshold_v1_2.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "[SAVED]"
    )

    print(
        TABLE_DIR
        / "model_threshold_v1_2_scan.csv"
    )

    print(
        TABLE_DIR
        / "model_threshold_v1_2.json"
    )

    print()
    print(
        "=" * 72
    )

    print(
        "STEP 3 EXPERIMENT COMPLETE"
    )

    print(
        "held-out test set과 production 모델은 "
        "아직 변경하지 않았습니다."
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()