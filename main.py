"""Adult Income 프로젝트의 개발·검증용 CLI 진입점.

웹서비스의 사용자 요청 처리는 향후 app.py가 담당한다.

이 파일은 개발 과정에서 다음 작업을 실행하기 위한 보조 도구다.

    data:
        원본 데이터를 로딩·정제하고 데이터 크기를 확인한다.

    eda:
        행 제거 전 데이터의 품질과 기본 분포를 확인한다.

    model:
        고소득 예측 모델을 학습·평가하고 모델 bundle을 저장한다.

    model-viz:
        저장된 모델 평가 결과로 개발용 진단 그래프를 생성한다.

    all:
        EDA → 모델 학습 → 모델 진단을 순서대로 실행한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import (
    RAW_DATA_PATH,
    ensure_directories,
)
from src.data import load_and_clean
from src.eda import run_eda
from src.modeling import train_income_model
from src.model_visualization import (create_model_visualizations,)


# ============================================================
# CLI
# ============================================================

def main() -> None:
    """개발·검증 단계의 명령줄 작업을 실행한다."""

    parser = argparse.ArgumentParser(
        description=(
            "Adult Income 분석·예측 서비스 "
            "개발 및 모델 검증 도구"
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=RAW_DATA_PATH,
        help=(
            "Adult CSV 경로 "
            f"(기본값: {RAW_DATA_PATH})"
        ),
    )

    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "data",
            "eda",
            "model",
            "model-viz",
        ],
        default="all",
        help="실행할 개발 단계",
    )

    args = parser.parse_args()
    ensure_directories()

    # --------------------------------------------------------
    # EDA
    # --------------------------------------------------------

    if args.stage in {
        "all",
        "eda",
    }:
        eda_result = run_eda(data_path=args.data,)
        summary = (eda_result["summary"])

        print("\n[EDA]")
        print(f"rows: {summary['rows']}")
        print(f"columns: {summary['columns']}")

        print(
            "missing cells: "
            f"{summary['missing_cells']}"
        )

        print(
            "duplicate rows: "
            f"{summary['duplicate_rows']}"
        )

        print(
            "high-income rate: "
            f"{summary['target']['high_income_rate']:.4f}"
        )

    # model-viz는 이미 저장된 모델 평가 결과만 사용하므로
    # Adult 데이터를 다시 불러올 필요가 없다.
    if args.stage == "model-viz":
        output_paths = (create_model_visualizations())
        print("\n[MODEL VISUALIZATION]")

        for (
            name,
            path,
        ) in output_paths.items():
            print(f"{name}: {path}")

        return

    # EDA 단독 실행은 정제 데이터가 필요하지 않다.
    if args.stage == "eda":
        return

    # --------------------------------------------------------
    # 공통 정제 데이터
    # --------------------------------------------------------

    df = load_and_clean(
        args.data,
        save_output=False,
    )

    if args.stage == "data":
        print("\n[DATA]")
        print(f"rows: {len(df)}")
        print(f"columns: {len(df.columns)}")

        return

    # --------------------------------------------------------
    # 모델 학습
    # --------------------------------------------------------

    if args.stage in {
        "all",
        "model",
    }:
        metrics = (train_income_model(df))
        print("\n[MODEL]")

        for (
            metric,
            value,
        ) in metrics.items():
            print(f"{metric}: {value}")

    # --------------------------------------------------------
    # 모델 진단 시각화
    # --------------------------------------------------------

    if args.stage == "all":
        output_paths = (create_model_visualizations())
        print("\n[MODEL VISUALIZATION]")

        for (
            name,
            path,
        ) in output_paths.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()