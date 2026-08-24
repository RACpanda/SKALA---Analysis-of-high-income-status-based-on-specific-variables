"""src.modeling의 v1.4 학습·저장·재로딩 계약을 합성 데이터로 검증한다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.modeling as modeling
from src.modeling import ModelingError


def _fast_feature_importance(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """단위 테스트 속도를 위해 persistence 계약에 필요한 형식만 반환한다."""

    return pd.DataFrame(
        {
            "feature": list(X_test.columns),
            "importance_mean": np.zeros(len(X_test.columns), dtype=float),
            "importance_std": np.zeros(len(X_test.columns), dtype=float),
        }
    )


def test_train_save_and_reload_model_contract(model_frame, monkeypatch):
    # 실제 v1.4 모델 성능은 별도 회귀 검증에서 확인했으므로,
    # 여기서는 permutation importance 반복 연산만 줄이고 학습/보정/저장은 실제 코드를 사용한다.
    monkeypatch.setattr(
        modeling,
        "_feature_importance",
        _fast_feature_importance,
    )

    evaluation = modeling.evaluate_income_model(model_frame)
    modeling._save_outputs(evaluation)

    expected_metric_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "brier_score",
        "log_loss",
        "actual_positive_rate",
        "mean_predicted_probability",
        "mean_probability_bias",
        "test_rows",
    }
    assert expected_metric_keys.issubset(evaluation.metrics)

    assert modeling.MODEL_BUNDLE_PATH.exists()
    for filename in [
        "model_metrics.json",
        "model_card.json",
        "model_input_schema.json",
        "model_fairness_by_group.csv",
        "model_feature_importance.csv",
        "model_predictions.csv",
    ]:
        assert (modeling.TABLE_DIR / filename).exists()

    feature_columns = evaluation.input_schema["feature_columns"]
    assert feature_columns == list(modeling.PREDICTION_FEATURE_COLUMNS)
    assert not {"income", "education-num", "fnlwgt"}.intersection(feature_columns)

    raw_input = model_frame[modeling.PREDICTION_FEATURE_COLUMNS].head(25).copy()
    prepared_input = modeling._coerce_features(
        raw_input,
        evaluation.input_schema,
    )
    original_probability = evaluation.pipeline.predict_proba(prepared_input)[:, 1]

    reloaded = modeling.predict_income(raw_input)

    assert set(reloaded.columns) == {"prediction", "probability"}
    assert np.allclose(
        original_probability,
        reloaded["probability"].to_numpy(),
    )
    assert reloaded["probability"].between(0, 1).all()


def test_model_rejects_missing_required_prediction_feature(model_frame):
    df = model_frame.drop(columns=["occupation"])

    with pytest.raises(ModelingError, match="모델 학습에 필요한 변수가 없습니다"):
        modeling.evaluate_income_model(df)
