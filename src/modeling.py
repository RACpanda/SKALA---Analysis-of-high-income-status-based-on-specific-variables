"""Adult Census Income 고소득 예측 모델.

이 모듈은 연관성 분석을 수행하지 않는다.

주요 역할:
    1. 고소득 여부 예측 모델 학습·평가
    2. 학습된 모델과 입력 스키마 저장
    3. 사용자 입력의 고소득 확률 예측
    4. 개인 예측에 대한 모델 기반 설명 생성
    5. 입력값 변화에 따른 What-if 예측 생성

주의:
    모델의 예측 확률, feature importance, 개인별 feature impact,
    What-if 결과는 모두 머신러닝 모델의 예측 결과이다.
    변수의 인과효과를 의미하지 않는다.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    ANALYSIS_VARIABLE_TYPES,
    MODEL_DIR,
    PREDICTION_FEATURE_COLUMNS,
    RANDOM_STATE,
    TABLE_DIR,
    TARGET_COLUMN,
    ensure_directories,
)


logger = logging.getLogger(__name__)


# ============================================================
# 모델 설정
# ============================================================

MODEL_BUNDLE_PATH = (
    MODEL_DIR
    / "income_model_bundle.joblib"
)

FAIRNESS_GROUP_COLUMNS = [
    "sex",
    "race",
]

MIN_SAMPLES_PER_CLASS = 10
MIN_RELIABLE_GROUP_POSITIVES = 30

FAIRNESS_MIN_GROUP_RECALL = 0.60
FAIRNESS_MAX_RECALL_GAP = 0.10


MODEL_PARAMS = {
    "learning_rate": (
        0.14447746112718687
    ),
    "max_depth": 5,
    "max_iter": 154,
    "l2_regularization": (
        0.45606998421703593
    ),
}


class ModelingError(RuntimeError):
    """모델 학습·저장·추론 과정에서 발생하는 오류."""


@dataclass
class ModelEvaluation:
    """모델 학습·평가 결과를 메모리에서 전달하는 컨테이너."""

    pipeline: Pipeline
    metrics: dict
    fairness: pd.DataFrame
    feature_importance: pd.DataFrame
    predictions: pd.DataFrame
    input_schema: dict
    model_card: dict = field(
        default_factory=dict
    )


# ============================================================
# 피처 설정
# ============================================================

def _feature_type_columns(
) -> tuple[
    list[str],
    list[str],
]:
    """config.py 기준으로 예측 피처를 수치형과 범주형으로 구분한다."""

    unknown_features = [
        column
        for column in PREDICTION_FEATURE_COLUMNS
        if column
        not in ANALYSIS_VARIABLE_TYPES
    ]

    if unknown_features:
        raise ModelingError(
            "예측 변수의 타입 설정이 없습니다: "
            f"{unknown_features}"
        )

    numeric_columns = [
        column
        for column in PREDICTION_FEATURE_COLUMNS
        if ANALYSIS_VARIABLE_TYPES[
            column
        ]
        == "continuous"
    ]

    categorical_columns = [
        column
        for column in PREDICTION_FEATURE_COLUMNS
        if ANALYSIS_VARIABLE_TYPES[
            column
        ]
        in {
            "binary",
            "categorical",
        }
    ]

    return (
        numeric_columns,
        categorical_columns,
    )


# ============================================================
# 학습 데이터 검증
# ============================================================

def _prepare_training_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """모델 학습에 사용할 행과 열 구조를 검증한다."""

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise ModelingError(
            "모델 학습 입력은 "
            "pandas.DataFrame이어야 합니다."
        )

    if df.empty:
        raise ModelingError(
            "모델 학습용 데이터가 비어 있습니다."
        )

    required_columns = [
        TARGET_COLUMN,
        *PREDICTION_FEATURE_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ModelingError(
            "모델 학습에 필요한 변수가 없습니다: "
            f"{missing_columns}"
        )

    # 결과변수가 없는 행은 지도학습에 사용할 수 없으므로
    # 학습 단계에서만 제거한다.
    model_df = (
        df
        .dropna(
            subset=[
                TARGET_COLUMN,
            ]
        )
        .copy()
        .reset_index(drop=True)
    )

    if model_df.empty:
        raise ModelingError(
            "target 결측값을 제외한 뒤 "
            "학습 가능한 표본이 없습니다."
        )

    try:
        target = pd.to_numeric(
            model_df[
                TARGET_COLUMN
            ],
            errors="raise",
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ModelingError(
            f"target '{TARGET_COLUMN}'을 "
            "숫자로 해석할 수 없습니다."
        ) from exc

    target_values = set(
        target.unique()
    )

    if target_values != {
        0,
        1,
    }:
        raise ModelingError(
            f"target '{TARGET_COLUMN}'은 "
            "0과 1을 모두 포함해야 합니다. "
            f"현재 값: {target_values}"
        )

    class_counts = (
        target
        .value_counts()
    )

    if (
        class_counts.min()
        < MIN_SAMPLES_PER_CLASS
    ):
        raise ModelingError(
            "target 클래스 중 표본이 너무 적습니다. "
            f"최소 클래스={int(class_counts.min())}, "
            f"필요 기준={MIN_SAMPLES_PER_CLASS}"
        )

    return model_df


# ============================================================
# 모델 입력 스키마
# ============================================================

def _build_input_schema(
    X: pd.DataFrame,
) -> dict:
    """예측 입력 UI와 추론 검증에 사용할 기본 스키마를 생성한다."""

    schema: dict = {
        "feature_columns": list(
            PREDICTION_FEATURE_COLUMNS
        ),
        "features": {},
    }

    for column in (
        PREDICTION_FEATURE_COLUMNS
    ):
        variable_type = (
            ANALYSIS_VARIABLE_TYPES[
                column
            ]
        )

        if variable_type == "continuous":
            numeric = pd.to_numeric(
                X[column],
                errors="coerce",
            )

            valid = (
                numeric
                .dropna()
            )

            if valid.empty:
                raise ModelingError(
                    f"수치형 변수 '{column}'에 "
                    "학습 가능한 값이 없습니다."
                )

            schema["features"][
                column
            ] = {
                "type": (
                    "continuous"
                ),
                "minimum": float(
                    valid.min()
                ),
                "maximum": float(
                    valid.max()
                ),
            }

        else:
            valid = (
                X[column]
                .dropna()
                .astype(object)
            )

            if valid.empty:
                raise ModelingError(
                    f"범주형 변수 '{column}'에 "
                    "학습 가능한 값이 없습니다."
                )

            levels = sorted(
                valid.unique().tolist(),
                key=lambda value: str(
                    value
                ),
            )

            schema["features"][
                column
            ] = {
                "type": variable_type,
                "levels": levels,
            }

    return schema


def _add_training_references(
    schema: dict,
    X_train: pd.DataFrame,
) -> dict:
    """개인 예측 설명과 What-if에 사용할 학습셋 대표값을 추가한다."""

    result = deepcopy(
        schema
    )

    for column in (
        PREDICTION_FEATURE_COLUMNS
    ):
        info = result[
            "features"
        ][column]

        if (
            info["type"]
            == "continuous"
        ):
            values = pd.to_numeric(
                X_train[column],
                errors="coerce",
            ).dropna()

            if values.empty:
                raise ModelingError(
                    f"'{column}'의 학습셋 대표값을 "
                    "계산할 수 없습니다."
                )

            info[
                "reference_value"
            ] = float(
                values.median()
            )

            info[
                "q05"
            ] = float(
                values.quantile(
                    0.05
                )
            )

            info[
                "q95"
            ] = float(
                values.quantile(
                    0.95
                )
            )

        else:
            values = (
                X_train[column]
                .dropna()
                .astype(object)
            )

            mode = (
                values.mode()
            )

            if mode.empty:
                raise ModelingError(
                    f"'{column}'의 학습셋 대표 범주를 "
                    "계산할 수 없습니다."
                )

            info[
                "reference_value"
            ] = mode.iloc[0]

    return result


# ============================================================
# 피처 dtype 통일
# ============================================================

def _coerce_features(
    X: pd.DataFrame,
    input_schema: dict,
    *,
    allow_unknown_categories: bool = False,
) -> pd.DataFrame:
    """학습·추론 데이터의 dtype을 모델 입력 스키마와 맞춘다."""

    expected_columns = (
        input_schema[
            "feature_columns"
        ]
    )

    missing_columns = [
        column
        for column in expected_columns
        if column not in X.columns
    ]

    if missing_columns:
        raise ModelingError(
            "예측에 필요한 입력 변수가 없습니다: "
            f"{missing_columns}"
        )

    result = (
        X[
            expected_columns
        ]
        .copy()
    )

    for column in expected_columns:
        info = (
            input_schema[
                "features"
            ][column]
        )

        if (
            info["type"]
            == "continuous"
        ):
            original = (
                result[column]
            )

            converted = pd.to_numeric(
                original,
                errors="coerce",
            )

            invalid_mask = (
                original.notna()
                & converted.isna()
            )

            if invalid_mask.any():
                invalid_values = (
                    original.loc[
                        invalid_mask
                    ]
                    .unique()
                    .tolist()
                )

                raise ModelingError(
                    f"수치형 변수 '{column}'에 "
                    "숫자로 변환할 수 없는 값이 있습니다: "
                    f"{invalid_values}"
                )

            result[column] = (
                converted.astype(
                    float
                )
            )

        else:
            levels = (
                info["levels"]
            )

            raw = (
                result[column]
                .astype(object)
                .where(
                    result[column]
                    .notna(),
                    np.nan,
                )
            )

            unknown_mask = (
                pd.Series(
                    raw,
                    index=result.index,
                )
                .notna()
                & ~pd.Series(
                    raw,
                    index=result.index,
                )
                .isin(levels)
            )

            if unknown_mask.any():
                if allow_unknown_categories:
                    raw = raw.mask(
                        unknown_mask,
                        np.nan,
                    )
                else:
                    unknown_values = (
                        raw.loc[
                            unknown_mask
                        ]
                        .unique()
                        .tolist()
                    )

                    raise ModelingError(
                        f"범주형 변수 '{column}'에 "
                        "학습 당시 존재하지 않은 값이 있습니다: "
                        f"{unknown_values}"
                    )

            result[column] = pd.Categorical(
                raw,
                categories=levels,
            )

    return result


# ============================================================
# sklearn Pipeline
# ============================================================

def _build_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    """전처리와 HistGradientBoosting 모델을 하나의 Pipeline으로 구성한다."""

    preprocessing = (
        ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="median"
                                ),
                            ),
                        ]
                    ),
                    numeric_columns,
                ),
                (
                    "categorical",
                    "passthrough",
                    categorical_columns,
                ),
            ]
        )
    )

    preprocessing.set_output(
        transform="pandas"
    )

    model = (
        HistGradientBoostingClassifier(
            categorical_features=(
                "from_dtype"
            ),
            class_weight="balanced",
            random_state=RANDOM_STATE,
            **MODEL_PARAMS,
        )
    )

    return Pipeline(
        [
            (
                "preprocessing",
                preprocessing,
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# 모델 공정성 진단
# ============================================================

def _fairness_by_group(
    df_test: pd.DataFrame,
    y_test: pd.Series,
    prediction: np.ndarray,
    group_columns: list[str],
) -> pd.DataFrame:
    """집단별 Recall과 False Negative Rate를 계산한다."""

    rows: list[dict] = []

    prediction_series = pd.Series(
        prediction,
        index=y_test.index,
    )

    for column in group_columns:
        if (
            column
            not in df_test.columns
        ):
            continue

        groups = (
            df_test
            .groupby(
                column,
                observed=True,
            )
            .groups
        )

        for (
            group_value,
            indices,
        ) in groups.items():
            group_y = (
                y_test.loc[
                    indices
                ]
            )

            group_prediction = (
                prediction_series.loc[
                    indices
                ]
            )

            positives = int(
                (
                    group_y == 1
                ).sum()
            )

            if positives == 0:
                continue

            recall = float(
                recall_score(
                    group_y,
                    group_prediction,
                    zero_division=0,
                )
            )

            rows.append(
                {
                    "group_column": (
                        column
                    ),
                    "group_value": (
                        group_value
                    ),
                    "n": int(
                        len(indices)
                    ),
                    "n_actual_positive": (
                        positives
                    ),
                    "recall": recall,
                    "false_negative_rate": (
                        float(
                            1
                            - recall
                        )
                    ),
                    "reliable": (
                        positives
                        >= MIN_RELIABLE_GROUP_POSITIVES
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def _assess_fairness(
    fairness: pd.DataFrame,
) -> dict:
    """충분한 표본을 가진 집단의 Recall 격차를 진단한다."""

    if fairness.empty:
        return {
            "status": "skipped",
            "reason": (
                "그룹별 평가 표본 없음"
            ),
        }

    reliable = (
        fairness.loc[
            fairness[
                "reliable"
            ]
        ]
    )

    if reliable.empty:
        return {
            "status": "skipped",
            "reason": (
                "신뢰 가능한 집단 없음"
            ),
        }

    violations: list[dict] = []

    for (
        column,
        group,
    ) in reliable.groupby(
        "group_column"
    ):
        minimum_row = (
            group.loc[
                group[
                    "recall"
                ].idxmin()
            ]
        )

        maximum_row = (
            group.loc[
                group[
                    "recall"
                ].idxmax()
            ]
        )

        minimum_recall = float(
            minimum_row[
                "recall"
            ]
        )

        maximum_recall = float(
            maximum_row[
                "recall"
            ]
        )

        recall_gap = (
            maximum_recall
            - minimum_recall
        )

        if (
            minimum_recall
            < FAIRNESS_MIN_GROUP_RECALL
        ):
            violations.append(
                {
                    "group_column": (
                        column
                    ),
                    "group_value": (
                        minimum_row[
                            "group_value"
                        ]
                    ),
                    "type": (
                        "min_recall"
                    ),
                    "value": (
                        minimum_recall
                    ),
                    "threshold": (
                        FAIRNESS_MIN_GROUP_RECALL
                    ),
                }
            )

        if (
            recall_gap
            > FAIRNESS_MAX_RECALL_GAP
        ):
            violations.append(
                {
                    "group_column": (
                        column
                    ),
                    "type": (
                        "recall_gap"
                    ),
                    "value": (
                        recall_gap
                    ),
                    "threshold": (
                        FAIRNESS_MAX_RECALL_GAP
                    ),
                    "lowest_group_value": (
                        minimum_row[
                            "group_value"
                        ]
                    ),
                    "lowest_group_recall": (
                        minimum_recall
                    ),
                    "highest_group_value": (
                        maximum_row[
                            "group_value"
                        ]
                    ),
                    "highest_group_recall": (
                        maximum_recall
                    ),
                }
            )

    status = (
        "fail"
        if violations
        else "pass"
    )

    for violation in violations:
        logger.warning(
            "[공정성 진단] %s: %s",
            violation[
                "group_column"
            ],
            violation,
        )

    return {
        "status": status,
        "min_recall_threshold": (
            FAIRNESS_MIN_GROUP_RECALL
        ),
        "max_recall_gap_threshold": (
            FAIRNESS_MAX_RECALL_GAP
        ),
        "violations": (
            violations
        ),
    }


# ============================================================
# 전체 모델 feature importance
# ============================================================

def _feature_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """테스트셋 기준 permutation importance를 계산한다."""

    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance = (
        pd.DataFrame(
            {
                "feature": (
                    X_test.columns
                ),
                "importance_mean": (
                    result.importances_mean
                ),
                "importance_std": (
                    result.importances_std
                ),
            }
        )
        .sort_values(
            "importance_mean",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return importance


# ============================================================
# 모델 메타데이터
# ============================================================

def _build_model_card(
    *,
    metrics: dict,
    train_rows: int,
    test_rows: int,
    fairness_assessment: dict,
) -> dict:
    """학습된 모델의 재현·진단 메타데이터를 생성한다."""

    return {
        "generated_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "model_name": (
            metrics[
                "model_name"
            ]
        ),
        "hyperparameters": (
            MODEL_PARAMS
        ),
        "random_state": (
            RANDOM_STATE
        ),
        "train_rows": (
            int(train_rows)
        ),
        "test_rows": (
            int(test_rows)
        ),
        "feature_columns": list(
            PREDICTION_FEATURE_COLUMNS
        ),
        "feature_types": {
            column: (
                ANALYSIS_VARIABLE_TYPES[
                    column
                ]
            )
            for column
            in PREDICTION_FEATURE_COLUMNS
        },
        "sklearn_version": (
            sklearn.__version__
        ),
        "python_version": (
            platform.python_version()
        ),
        "fairness_check": (
            fairness_assessment
        ),
    }


# ============================================================
# 모델 학습·평가
# ============================================================

def evaluate_income_model(
    df: pd.DataFrame,
) -> ModelEvaluation:
    """모델을 학습·평가하고 결과 객체를 반환한다."""

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

    # 입력 스키마와 대표값은 학습 데이터만 이용한다.
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

    X_train = _coerce_features(
        raw_X_train,
        input_schema,
    )

    X_test = _coerce_features(
        raw_X_test,
        input_schema,
        allow_unknown_categories=True,
    )

    (
        numeric_columns,
        categorical_columns,
    ) = _feature_type_columns()

    pipeline = _build_pipeline(
        numeric_columns,
        categorical_columns,
    )

    try:
        pipeline.fit(
            X_train,
            y_train,
        )
    except Exception as exc:
        raise ModelingError(
            f"모델 학습에 실패했습니다: {exc}"
        ) from exc

    try:
        prediction = (
            pipeline.predict(
                X_test
            )
        )

        probability = (
            pipeline.predict_proba(
                X_test
            )[:, 1]
        )

    except Exception as exc:
        raise ModelingError(
            f"테스트셋 예측에 실패했습니다: {exc}"
        ) from exc

    metrics = {
        "model_name": (
            "hist_gradient_boosting"
        ),
        "test_rows": int(
            len(X_test)
        ),
        "accuracy": float(
            accuracy_score(
                y_test,
                prediction,
            )
        ),
        "precision": float(
            precision_score(
                y_test,
                prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                prediction,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                prediction,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                probability,
            )
        ),
    }

    predictions = pd.DataFrame(
        {
            "row_id": (
                X_test.index
            ),
            "y_test": (
                y_test.to_numpy()
            ),
            "y_pred": (
                prediction
            ),
            "y_proba": (
                probability
            ),
        }
    )

    fairness = _fairness_by_group(
        X_test,
        y_test,
        prediction,
        FAIRNESS_GROUP_COLUMNS,
    )

    fairness_assessment = (
        _assess_fairness(
            fairness
        )
    )

    feature_importance = (
        _feature_importance(
            pipeline,
            X_test,
            y_test,
        )
    )

    model_card = (
        _build_model_card(
            metrics=metrics,
            train_rows=len(
                X_train
            ),
            test_rows=len(
                X_test
            ),
            fairness_assessment=(
                fairness_assessment
            ),
        )
    )

    return ModelEvaluation(
        pipeline=pipeline,
        metrics=metrics,
        fairness=fairness,
        feature_importance=(
            feature_importance
        ),
        predictions=predictions,
        input_schema=(
            input_schema
        ),
        model_card=model_card,
    )


# ============================================================
# 모델 저장
# ============================================================

def _save_outputs(
    evaluation: ModelEvaluation,
) -> None:
    """배포용 모델 bundle과 개발용 평가 산출물을 저장한다."""

    ensure_directories()

    bundle = {
        "pipeline": (
            evaluation.pipeline
        ),
        "input_schema": (
            evaluation.input_schema
        ),
        "model_card": (
            evaluation.model_card
        ),
        "global_feature_importance": (
            evaluation.feature_importance
            .to_dict(
                orient="records"
            )
        ),
    }

    try:
        joblib.dump(
            bundle,
            MODEL_BUNDLE_PATH,
        )

        (
            TABLE_DIR
            / "model_metrics.json"
        ).write_text(
            json.dumps(
                evaluation.metrics,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        (
            TABLE_DIR
            / "model_card.json"
        ).write_text(
            json.dumps(
                evaluation.model_card,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        (
            TABLE_DIR
            / "model_input_schema.json"
        ).write_text(
            json.dumps(
                evaluation.input_schema,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        evaluation.fairness.to_csv(
            TABLE_DIR
            / "model_fairness_by_group.csv",
            index=False,
        )

        evaluation.feature_importance.to_csv(
            TABLE_DIR
            / "model_feature_importance.csv",
            index=False,
        )

        evaluation.predictions.to_csv(
            TABLE_DIR
            / "model_predictions.csv",
            index=False,
        )

    except (
        OSError,
        TypeError,
    ) as exc:
        raise ModelingError(
            f"모델 산출물 저장에 실패했습니다: {exc}"
        ) from exc


def train_income_model(
    df: pd.DataFrame,
) -> dict:
    """고소득 예측 모델을 학습·평가하고 배포용 bundle을 저장한다."""

    evaluation = (
        evaluate_income_model(
            df
        )
    )

    _save_outputs(
        evaluation
    )

    return evaluation.metrics


# ============================================================
# 저장된 모델 로딩
# ============================================================

def _load_model_bundle() -> dict:
    """저장된 예측 모델과 입력 스키마를 불러온다."""

    if not MODEL_BUNDLE_PATH.exists():
        raise ModelingError(
            "학습된 모델 bundle이 없습니다: "
            f"{MODEL_BUNDLE_PATH}. "
            "train_income_model()을 먼저 실행하세요."
        )

    try:
        bundle = joblib.load(
            MODEL_BUNDLE_PATH
        )
    except Exception as exc:
        raise ModelingError(
            f"모델 bundle을 불러오지 못했습니다: {exc}"
        ) from exc

    if not isinstance(
        bundle,
        dict,
    ):
        raise ModelingError(
            "저장된 모델 bundle 형식이 올바르지 않습니다."
        )

    required_keys = {
        "pipeline",
        "input_schema",
        "model_card",
        "global_feature_importance",
    }

    missing_keys = (
        required_keys
        - set(bundle)
    )

    if missing_keys:
        raise ModelingError(
            "모델 bundle에 필요한 항목이 없습니다: "
            f"{sorted(missing_keys)}"
        )

    return bundle


def get_global_feature_importance(
) -> pd.DataFrame:
    """현재 배포 모델의 전체 테스트셋 permutation importance를 반환한다."""

    bundle = (
        _load_model_bundle()
    )

    return pd.DataFrame(
        bundle[
            "global_feature_importance"
        ]
    )


def get_prediction_input_schema() -> dict:
    """웹 UI가 입력 폼을 구성할 수 있도록 학습 당시 입력 스키마를 반환한다."""

    bundle = (
        _load_model_bundle()
    )

    return deepcopy(
        bundle[
            "input_schema"
        ]
    )


# ============================================================
# 공통 추론
# ============================================================

def _prepare_prediction_frame(
    df: pd.DataFrame,
    bundle: dict,
) -> pd.DataFrame:
    """새 입력을 학습 당시 feature schema와 동일하게 변환한다."""

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise ModelingError(
            "예측 입력은 pandas.DataFrame이어야 합니다."
        )

    if df.empty:
        raise ModelingError(
            "예측할 데이터가 비어 있습니다."
        )

    return _coerce_features(
        df,
        bundle[
            "input_schema"
        ],
    )


def _predict_with_bundle(
    df: pd.DataFrame,
    bundle: dict,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """이미 로드된 bundle을 이용해 분류값과 고소득 확률을 계산한다."""

    X = _prepare_prediction_frame(
        df,
        bundle,
    )

    pipeline: Pipeline = (
        bundle[
            "pipeline"
        ]
    )

    try:
        prediction = (
            pipeline.predict(
                X
            )
        )

        probability = (
            pipeline.predict_proba(
                X
            )[:, 1]
        )

    except Exception as exc:
        raise ModelingError(
            f"고소득 예측에 실패했습니다: {exc}"
        ) from exc

    return (
        prediction,
        probability,
    )


def predict_income(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """여러 입력 행에 대해 고소득 분류값과 확률을 반환한다."""

    bundle = (
        _load_model_bundle()
    )

    (
        prediction,
        probability,
    ) = _predict_with_bundle(
        df,
        bundle,
    )

    return pd.DataFrame(
        {
            "prediction": (
                prediction.astype(
                    int
                )
            ),
            "probability": (
                probability
            ),
        },
        index=df.index,
    )


# ============================================================
# 개인 예측 설명
# ============================================================

def _validate_user_input(
    user_input: dict,
    input_schema: dict,
) -> None:
    """홈페이지에서 받은 한 사람의 입력 key를 검증한다."""

    if not isinstance(
        user_input,
        dict,
    ):
        raise ModelingError(
            "user_input은 dict 형태여야 합니다."
        )

    if not user_input:
        raise ModelingError(
            "예측 입력값이 없습니다."
        )

    expected = set(
        input_schema[
            "feature_columns"
        ]
    )

    provided = set(
        user_input
    )

    missing = sorted(
        expected
        - provided
    )

    unexpected = sorted(
        provided
        - expected
    )

    if missing:
        raise ModelingError(
            "예측에 필요한 입력 변수가 없습니다: "
            f"{missing}"
        )

    if unexpected:
        raise ModelingError(
            "예측 모델에서 사용하지 않는 입력 변수가 있습니다: "
            f"{unexpected}"
        )


def _local_feature_impacts(
    user_input: dict,
    bundle: dict,
    original_probability: float,
) -> pd.DataFrame:
    """각 feature를 학습셋 대표값으로 하나씩 바꾸어 예측 변화량을 계산한다.

    impact가 양수이면 현재 입력값이 해당 feature의 대표값보다
    모델의 고소득 예측 확률을 높이는 방향이고,
    음수이면 낮추는 방향이다.

    각 feature를 독립적으로 한 번씩 바꾸는 방식이므로
    impact 값들의 합은 전체 예측 확률과 일치하지 않는다.
    """

    schema = (
        bundle[
            "input_schema"
        ]
    )

    rows: list[dict] = []

    for feature in (
        schema[
            "feature_columns"
        ]
    ):
        reference_value = (
            schema[
                "features"
            ][feature][
                "reference_value"
            ]
        )

        modified_input = dict(
            user_input
        )

        modified_input[
            feature
        ] = reference_value

        modified_df = pd.DataFrame(
            [
                modified_input
            ]
        )

        (
            _,
            baseline_probability,
        ) = _predict_with_bundle(
            modified_df,
            bundle,
        )

        baseline_probability_value = (
            float(
                baseline_probability[
                    0
                ]
            )
        )

        impact = float(
            original_probability
            - baseline_probability_value
        )

        rows.append(
            {
                "feature": (
                    feature
                ),
                "current_value": (
                    user_input[
                        feature
                    ]
                ),
                "reference_value": (
                    reference_value
                ),
                "original_probability": (
                    original_probability
                ),
                "reference_probability": (
                    baseline_probability_value
                ),
                "impact_probability": (
                    impact
                ),
                "impact_percentage_points": (
                    impact
                    * 100
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .assign(
            absolute_impact=lambda frame: (
                frame[
                    "impact_probability"
                ]
                .abs()
            )
        )
        .sort_values(
            "absolute_impact",
            ascending=False,
        )
        .drop(
            columns=[
                "absolute_impact",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def predict_income_input(
    user_input: dict,
) -> dict:
    """한 사람의 입력값으로 고소득 확률과 개인별 예측 설명을 반환한다."""

    bundle = (
        _load_model_bundle()
    )

    input_schema = (
        bundle[
            "input_schema"
        ]
    )

    _validate_user_input(
        user_input,
        input_schema,
    )

    input_df = pd.DataFrame(
        [
            user_input
        ]
    )

    (
        prediction,
        probability,
    ) = _predict_with_bundle(
        input_df,
        bundle,
    )

    predicted_class = int(
        prediction[
            0
        ]
    )

    probability_value = float(
        probability[
            0
        ]
    )

    explanation = (
        _local_feature_impacts(
            user_input,
            bundle,
            probability_value,
        )
    )

    model_card = (
        bundle[
            "model_card"
        ]
    )

    return {
        "input": dict(
            user_input
        ),

        "prediction": {
            "predicted_class": (
                predicted_class
            ),
            "prediction_label": (
                ">50K"
                if predicted_class == 1
                else "<=50K"
            ),
            "high_income_probability": (
                probability_value
            ),
        },

        "explanation": {
            "method": (
                "single_feature_baseline_perturbation"
            ),
            "features": (
                explanation
                .to_dict(
                    orient="records"
                )
            ),
            "interpretation_note": (
                "각 변수의 현재 값을 학습 데이터의 대표값으로 "
                "하나씩 바꾸었을 때 예측 확률이 얼마나 변하는지 "
                "비교한 모델 기반 설명입니다. "
                "각 영향값은 서로 더할 수 없으며 인과효과를 의미하지 않습니다."
            ),
        },

        "model": {
            "model_name": (
                model_card.get(
                    "model_name"
                )
            ),
            "trained_at": (
                model_card.get(
                    "generated_at"
                )
            ),
        },

        "interpretation_note": (
            "고소득 확률은 학습된 머신러닝 모델이 "
            "현재 입력 조건에 대해 출력한 예측값이며 "
            "실제 소득이나 개별 변수의 인과효과를 의미하지 않습니다."
        ),
    }


# ============================================================
# What-if Simulation
# ============================================================

def simulate_income_what_if(
    user_input: dict,
    feature: str,
    *,
    values: list | None = None,
    points: int = 15,
) -> pd.DataFrame:
    """다른 입력은 유지하고 하나의 feature만 변경한 예측 결과를 생성한다.

    연속형 변수:
        values가 없으면 학습셋 5~95 분위 범위에서 값을 생성한다.

    범주형 변수:
        values가 없으면 학습 당시 관측된 범주 전체를 사용한다.

    Returns:
        feature 값별 고소득 예측 확률 DataFrame.
    """

    bundle = (
        _load_model_bundle()
    )

    schema = (
        bundle[
            "input_schema"
        ]
    )

    _validate_user_input(
        user_input,
        schema,
    )

    if (
        feature
        not in schema[
            "feature_columns"
        ]
    ):
        raise ModelingError(
            f"예측 모델에서 사용하지 않는 변수입니다: {feature}"
        )

    if points < 2:
        raise ModelingError(
            "points는 2 이상이어야 합니다."
        )

    info = (
        schema[
            "features"
        ][feature]
    )

    if values is None:
        if (
            info[
                "type"
            ]
            == "continuous"
        ):
            lower = float(
                info[
                    "q05"
                ]
            )

            upper = float(
                info[
                    "q95"
                ]
            )

            if lower == upper:
                lower = float(
                    info[
                        "minimum"
                    ]
                )

                upper = float(
                    info[
                        "maximum"
                    ]
                )

            if lower == upper:
                scenario_values = [
                    lower
                ]
            else:
                scenario_values = (
                    np.linspace(
                        lower,
                        upper,
                        points,
                    )
                    .tolist()
                )

        else:
            scenario_values = list(
                info[
                    "levels"
                ]
            )

    else:
        if not values:
            raise ModelingError(
                "What-if values가 비어 있습니다."
            )

        scenario_values = list(
            values
        )

    scenarios: list[dict] = []

    for value in scenario_values:
        scenario = dict(
            user_input
        )

        scenario[
            feature
        ] = value

        scenarios.append(
            scenario
        )

    scenario_df = pd.DataFrame(
        scenarios
    )

    (
        _,
        probabilities,
    ) = _predict_with_bundle(
        scenario_df,
        bundle,
    )

    return pd.DataFrame(
        {
            "feature": (
                feature
            ),
            "value": (
                scenario_values
            ),
            "high_income_probability": (
                probabilities
            ),
            "high_income_probability_percent": (
                probabilities
                * 100
            ),
        }
    )