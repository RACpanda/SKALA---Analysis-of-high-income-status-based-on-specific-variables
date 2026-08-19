"""Version 1.2 모델 재튜닝 실험.

현재 프로덕션 src/modeling.py의 데이터 계약과 Pipeline을 그대로 사용하여
HistGradientBoostingClassifier의 하이퍼파라미터를 다시 탐색한다.

중요:
    - held-out test set은 이번 단계에서 평가하지 않는다.
    - 모델 선택은 train set 내부 5-fold CV ROC-AUC로만 수행한다.
    - Calibration / Threshold 작업까지 끝난 뒤 test set을 최종 평가한다.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.stats import (
    loguniform,
    randint,
    uniform,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from src.config import (
    PREDICTION_FEATURE_COLUMNS,
    RANDOM_STATE,
    TABLE_DIR,
    TARGET_COLUMN,
    ensure_directories,
)
from src.data import (
    load_and_clean,
)
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

CV_FOLDS = 5
N_ITER = 40


PARAM_DISTRIBUTIONS = {
    "model__learning_rate": (
        loguniform(
            0.03,
            0.25,
        )
    ),
    "model__max_iter": (
        randint(
            100,
            351,
        )
    ),
    "model__max_depth": [
        None,
        3,
        5,
        7,
        9,
    ],
    "model__max_leaf_nodes": [
        15,
        31,
        63,
        None,
    ],
    "model__min_samples_leaf": (
        randint(
            10,
            51,
        )
    ),
    "model__l2_regularization": (
        uniform(
            0.0,
            2.0,
        )
    ),
}


# ============================================================
# JSON 변환
# ============================================================

def _jsonable(
    value,
):
    """numpy 값을 JSON 저장 가능한 Python 값으로 변환한다."""

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        dict,
    ):
        return {
            key: _jsonable(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            _jsonable(
                item
            )
            for item in value
        ]

    return value


# ============================================================
# 출력
# ============================================================

def _print_cv_result(
    name: str,
    scores: np.ndarray,
) -> None:
    """CV ROC-AUC 결과를 출력한다."""

    print(
        f"{name}"
    )

    print(
        "  fold scores:",
        ", ".join(
            f"{score:.6f}"
            for score in scores
        ),
    )

    print(
        f"  mean: {scores.mean():.6f}"
    )

    print(
        f"  std : {scores.std(ddof=1):.6f}"
    )


# ============================================================
# 실험
# ============================================================

def main() -> None:
    print(
        "=" * 70
    )

    print(
        "Adult Income Explorer"
    )

    print(
        "Version 1.2 - Model Retuning"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # 현재 서비스 데이터 사용
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
    # 현재 production과 동일한 80:20 split
    #
    # test set은 v1.2 마지막까지 사용하지 않는다.
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
        f"total rows : {len(raw_X):,}"
    )

    print(
        f"train rows : {len(raw_X_train):,}"
    )

    print(
        f"test rows  : {len(raw_X_test):,}"
    )

    print(
        "features   : "
        f"{len(PREDICTION_FEATURE_COLUMNS)}"
    )

    print()
    print(
        "held-out test set은 "
        "이번 단계에서 사용하지 않습니다."
    )

    # --------------------------------------------------------
    # production과 동일한 입력 스키마
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

    # --------------------------------------------------------
    # 동일 CV split
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # 현재 production baseline
    # --------------------------------------------------------

    baseline_pipeline = (
        _build_pipeline(
            numeric_columns,
            categorical_columns,
        )
    )

    print()
    print(
        "[BASELINE]"
    )

    print(
        "current MODEL_PARAMS:"
    )

    print(
        json.dumps(
            MODEL_PARAMS,
            indent=2,
        )
    )

    baseline_start = (
        time.perf_counter()
    )

    baseline_scores = (
        cross_val_score(
            baseline_pipeline,
            X_train,
            y_train,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
        )
    )

    baseline_seconds = (
        time.perf_counter()
        - baseline_start
    )

    _print_cv_result(
        "Current model",
        baseline_scores,
    )

    print(
        f"  time: {baseline_seconds:.2f}s"
    )

    # --------------------------------------------------------
    # RandomizedSearchCV
    # --------------------------------------------------------

    candidate_pipeline = (
        _build_pipeline(
            numeric_columns,
            categorical_columns,
        )
    )

    search = RandomizedSearchCV(
        estimator=(
            candidate_pipeline
        ),
        param_distributions=(
            PARAM_DISTRIBUTIONS
        ),
        n_iter=N_ITER,
        scoring="roc_auc",
        cv=cv,
        random_state=(
            RANDOM_STATE
        ),
        n_jobs=-1,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )

    print()
    print(
        "[TUNING]"
    )

    print(
        f"n_iter={N_ITER}, "
        f"cv={CV_FOLDS}"
    )

    tuning_start = (
        time.perf_counter()
    )

    search.fit(
        X_train,
        y_train,
    )

    tuning_seconds = (
        time.perf_counter()
        - tuning_start
    )

    # --------------------------------------------------------
    # 최적 후보의 fold별 score 추출
    # --------------------------------------------------------

    best_index = (
        search.best_index_
    )

    tuned_scores = np.asarray(
        [
            search.cv_results_[
                f"split{fold}_test_score"
            ][best_index]
            for fold in range(
                CV_FOLDS
            )
        ],
        dtype=float,
    )

    print()

    _print_cv_result(
        "Tuned candidate",
        tuned_scores,
    )

    print(
        f"  tuning time: "
        f"{tuning_seconds:.2f}s"
    )

    # --------------------------------------------------------
    # model__ prefix 제거
    # --------------------------------------------------------

    best_model_params = {
        key.replace(
            "model__",
            "",
        ): _jsonable(
            value
        )
        for key, value
        in search.best_params_.items()
    }

    print()
    print(
        "[BEST PARAMETERS]"
    )

    print(
        json.dumps(
            best_model_params,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # 현재 vs 후보 비교
    # --------------------------------------------------------

    baseline_mean = float(
        baseline_scores.mean()
    )

    tuned_mean = float(
        tuned_scores.mean()
    )

    difference = (
        tuned_mean
        - baseline_mean
    )

    print()
    print(
        "[COMPARISON]"
    )

    print(
        f"baseline CV ROC-AUC : "
        f"{baseline_mean:.6f}"
    )

    print(
        f"tuned CV ROC-AUC    : "
        f"{tuned_mean:.6f}"
    )

    print(
        f"difference          : "
        f"{difference:+.6f}"
    )

    # --------------------------------------------------------
    # 결과 저장
    # --------------------------------------------------------

    ensure_directories()

    summary = {
        "version": "1.2",
        "selection_metric": (
            "cv_roc_auc"
        ),
        "cv_folds": (
            CV_FOLDS
        ),
        "n_iter": (
            N_ITER
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
        "feature_columns": list(
            PREDICTION_FEATURE_COLUMNS
        ),
        "baseline": {
            "params": (
                _jsonable(
                    MODEL_PARAMS
                )
            ),
            "fold_scores": (
                baseline_scores.tolist()
            ),
            "cv_roc_auc_mean": (
                baseline_mean
            ),
            "cv_roc_auc_std": float(
                baseline_scores.std(
                    ddof=1
                )
            ),
        },
        "tuned_candidate": {
            "params": (
                best_model_params
            ),
            "fold_scores": (
                tuned_scores.tolist()
            ),
            "cv_roc_auc_mean": (
                tuned_mean
            ),
            "cv_roc_auc_std": float(
                tuned_scores.std(
                    ddof=1
                )
            ),
        },
        "cv_roc_auc_difference": (
            difference
        ),
        "tuning_seconds": (
            tuning_seconds
        ),
    }

    (
        TABLE_DIR
        / "model_tuning_v1_2.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cv_results = (
        pd.DataFrame(
            search.cv_results_
        )
        .sort_values(
            "rank_test_score"
        )
        .reset_index(
            drop=True
        )
    )

    cv_results.to_csv(
        TABLE_DIR
        / "model_tuning_v1_2_cv.csv",
        index=False,
    )

    print()
    print(
        "[SAVED]"
    )

    print(
        TABLE_DIR
        / "model_tuning_v1_2.json"
    )

    print(
        TABLE_DIR
        / "model_tuning_v1_2_cv.csv"
    )

    print()
    print(
        "=" * 70
    )

    print(
        "STEP 1 COMPLETE"
    )

    print(
        "아직 MODEL_PARAMS와 배포 모델은 "
        "변경하지 않았습니다."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()