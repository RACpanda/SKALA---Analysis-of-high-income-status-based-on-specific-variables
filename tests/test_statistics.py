"""src.statistics의 v1.4 이진 집단 비교와 PSM 계약을 검증한다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.statistics import (
    binary_group_association,
    propensity_score_matching,
)


def test_binary_group_association_detects_clear_rate_gap():
    df = pd.DataFrame(
        {
            "sex": ["Female"] * 20 + ["Male"] * 20,
            "high_income": [1] * 4 + [0] * 16 + [1] * 16 + [0] * 4,
        }
    )

    result = binary_group_association(df, exposure="sex")
    analysis = result["analysis"]

    assert analysis["exposure_metadata"] == {
        "reference_level": "Female",
        "comparison_level": "Male",
    }
    assert analysis["comparison_rate"] > analysis["reference_rate"]
    assert analysis["rate_difference"] > 0
    assert analysis["fisher_exact_p_value"] < 0.05
    assert analysis["odds_ratio"] > 1


def test_binary_group_association_reports_zero_gap_for_identical_rates():
    df = pd.DataFrame(
        {
            "sex": ["Female"] * 20 + ["Male"] * 20,
            "high_income": ([1] * 10 + [0] * 10) * 2,
        }
    )

    analysis = binary_group_association(
        df,
        exposure="sex",
    )["analysis"]

    assert analysis["rate_difference"] == 0
    assert analysis["fisher_exact_p_value"] == 1.0


def test_propensity_score_matching_returns_pairs_and_improves_balance(
    association_frame,
):
    matched, balance, result = propensity_score_matching(
        association_frame,
        exposure="sex",
        covariates=["age", "race"],
        outcome="high_income",
    )

    matched_pairs = result["matching"]["matched_pairs"]

    assert matched_pairs >= 2
    assert len(matched) == matched_pairs * 2
    assert set(matched["matched_role"]) == {"comparison", "reference"}
    assert matched.groupby("pair_id").size().eq(2).all()

    # 현재 구현은 replacement를 사용하지 않으므로 기준집단 원본 행은 중복되지 않는다.
    reference_sources = matched.loc[
        matched["matched_role"] == "reference",
        "source_index",
    ]
    assert reference_sources.is_unique

    assert {"covariate", "smd_before", "smd_after"}.issubset(balance.columns)
    assert result["balance"]["max_smd_after"] < result["balance"]["max_smd_before"]
    assert result["balance"]["max_smd_after"] < 0.1

    # 실제 매칭 거리는 계산된 caliper 안에 있어야 한다.
    comparison_distances = matched.loc[
        matched["matched_role"] == "comparison",
        "match_distance",
    ].to_numpy(dtype=float)
    assert np.all(
        comparison_distances
        <= result["matching"]["caliper"] + 1e-12
    )
