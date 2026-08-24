# -*- coding: utf-8 -*-
"""
기준(베이스라인) 모델: 학습 데이터의 delivery_days 평균을
모든 테스트 주문에 동일하게 적용하고, 테스트 데이터에서
MAE / RMSE / R^2을 평가한다.

입력: data/processed/train.csv, data/processed/test.csv
      (analysis/04_train_test_split.py 에서 생성한 80/20 분할, random_state=42)
행 제외/값 대체 없음.

필요 라이브러리: pandas, numpy, scikit-learn (표준 라이브러리: sys, pathlib)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

sys.stdout.reconfigure(encoding="utf-8")

TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
OUT_DIR = Path("outputs/tables")


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    baseline_pred = train_df["delivery_days"].mean()
    y_test = test_df["delivery_days"].values
    y_pred = np.full_like(y_test, fill_value=baseline_pred, dtype=float)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    r2 = r2_score(y_test, y_pred)

    print("=" * 78)
    print("기준 모델: 예측값 = 학습 데이터 delivery_days 평균 (모든 테스트 주문 동일)")
    print("=" * 78)
    print(f"학습 데이터 평균(예측값으로 사용) : {baseline_pred:.3f} 일")
    print(f"테스트 데이터 행 수                : {len(test_df):,}")
    print()
    print(f"MAE  : {mae:.3f} 일")
    print(f"RMSE : {rmse:.3f} 일")
    print(f"R^2  : {r2:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "05_baseline_metrics.csv"
    pd.DataFrame(
        [
            {"지표": "baseline_prediction(train mean)", "값": round(baseline_pred, 3)},
            {"지표": "MAE", "값": round(mae, 3)},
            {"지표": "RMSE", "값": round(rmse, 3)},
            {"지표": "R2", "값": round(r2, 4)},
        ]
    ).to_csv(out_path, index=False, encoding="utf-8-sig")
    print()
    print(f"저장: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
