"""v1.4 테스트 공통 fixture.

실제 프로젝트의 outputs/ 산출물을 합성 데이터 테스트가 덮어쓰지 않도록
각 테스트마다 tmp_path 하위의 tables/models/figures 경로를 사용한다.

일부 모듈은 import 시점에 TABLE_DIR/MODEL_DIR에서 파생 경로를 미리 바인딩하므로
디렉터리 상수뿐 아니라 MODEL_BUNDLE_PATH, MODEL_METRICS_PATH,
MODEL_PREDICTIONS_PATH도 함께 패치한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.model_visualization as model_visualization
import src.modeling as modeling


@pytest.fixture(autouse=True)
def isolate_output_paths(tmp_path, monkeypatch):
    table_dir = tmp_path / "tables"
    model_dir = tmp_path / "models"
    figure_dir = tmp_path / "figures"

    for directory in (table_dir, model_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(modeling, "TABLE_DIR", table_dir)
    monkeypatch.setattr(modeling, "MODEL_DIR", model_dir)
    monkeypatch.setattr(
        modeling,
        "MODEL_BUNDLE_PATH",
        model_dir / "income_model_bundle.joblib",
    )

    monkeypatch.setattr(model_visualization, "TABLE_DIR", table_dir)
    monkeypatch.setattr(model_visualization, "FIGURE_DIR", figure_dir)
    monkeypatch.setattr(
        model_visualization,
        "MODEL_METRICS_PATH",
        table_dir / "model_metrics.json",
    )
    monkeypatch.setattr(
        model_visualization,
        "MODEL_PREDICTIONS_PATH",
        table_dir / "model_predictions.csv",
    )


@pytest.fixture
def association_frame() -> pd.DataFrame:
    """Continuous/Binary/Categorical/PSM을 함께 검증할 재현 가능한 합성 데이터."""

    rng = np.random.default_rng(42)
    n = 320

    age = np.clip(rng.normal(40, 10, n), 18, 70)
    p_male = 1 / (1 + np.exp(-((age - 40) / 10)))
    sex = np.where(rng.random(n) < p_male, "Male", "Female")

    race = rng.choice(
        ["White", "Black", "Asian-Pac-Islander"],
        size=n,
        p=[0.65, 0.20, 0.15],
    )
    education = rng.choice(
        ["HS-grad", "Bachelors", "Masters"],
        size=n,
        p=[0.50, 0.35, 0.15],
    )
    hours = np.clip(
        rng.normal(40 + (sex == "Male") * 2, 8, n),
        10,
        70,
    )

    logit = (
        -1.6
        + 0.035 * (age - 40)
        + 0.55 * (sex == "Male")
        + 0.45 * (education == "Bachelors")
        + 0.80 * (education == "Masters")
        + 0.015 * (hours - 40)
    )
    probability = 1 / (1 + np.exp(-logit))
    high_income = (rng.random(n) < probability).astype(int)

    return pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "race": race,
            "education": education,
            "hours-per-week": hours,
            "high_income": high_income,
        }
    )


@pytest.fixture
def model_frame() -> pd.DataFrame:
    """현재 12개 예측 피처 계약을 모두 포함하는 모델 학습용 합성 데이터."""

    rng = np.random.default_rng(7)
    n = 240

    age = rng.integers(20, 66, size=n)
    sex = rng.choice(["Male", "Female"], size=n)
    education = rng.choice(
        ["HS-grad", "Bachelors", "Masters"],
        size=n,
        p=[0.55, 0.35, 0.10],
    )
    hours = np.clip(
        np.rint(rng.normal(40, 8, size=n)),
        10,
        70,
    ).astype(int)
    capital_gain = np.where(
        rng.random(n) < 0.12,
        rng.integers(500, 8000, size=n),
        0,
    )
    capital_loss = np.where(
        rng.random(n) < 0.08,
        rng.integers(200, 2500, size=n),
        0,
    )

    workclass = rng.choice(
        ["Private", "Self-emp-not-inc", "Local-gov"],
        size=n,
    )
    marital_status = rng.choice(
        ["Never-married", "Married-civ-spouse", "Divorced"],
        size=n,
    )
    occupation = rng.choice(
        ["Adm-clerical", "Exec-managerial", "Sales"],
        size=n,
    )
    relationship = rng.choice(
        ["Not-in-family", "Husband", "Wife"],
        size=n,
    )
    race = rng.choice(
        ["White", "Black", "Asian-Pac-Islander"],
        size=n,
        p=[0.70, 0.20, 0.10],
    )
    native_country = rng.choice(
        ["United-States", "Mexico", "Canada"],
        size=n,
        p=[0.80, 0.10, 0.10],
    )

    logit = (
        -2.0
        + 0.035 * (age - 40)
        + 0.45 * (sex == "Male")
        + 0.50 * (education == "Bachelors")
        + 0.90 * (education == "Masters")
        + 0.018 * (hours - 40)
        + 0.00012 * capital_gain
        - 0.00008 * capital_loss
    )
    probability = 1 / (1 + np.exp(-logit))
    high_income = (rng.random(n) < probability).astype(int)

    return pd.DataFrame(
        {
            "age": age,
            "workclass": workclass,
            "education": education,
            "marital-status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "sex": sex,
            "capital-gain": capital_gain,
            "capital-loss": capital_loss,
            "hours-per-week": hours,
            "native-country": native_country,
            "high_income": high_income,
            # 아래 세 열은 모델 입력에서 제외되어야 하는 열이다.
            "income": np.where(high_income == 1, ">50K", "<=50K"),
            "education-num": rng.integers(1, 16, size=n),
            "fnlwgt": rng.integers(10_000, 300_000, size=n),
        }
    )
