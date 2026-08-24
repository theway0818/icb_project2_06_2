# -*- coding: utf-8 -*-
"""
distance_km 단일 변수로 delivery_days(배송 소요일수)를 예측하는
단순 선형회귀.

- 학습/평가: data/processed/train.csv, test.csv
  (analysis/04_train_test_split.py 에서 만든 80/20 분할, random_state=42)
- 결측 처리: distance_km 결측은 "학습 데이터의 중앙값"으로 대체.
  학습·테스트 모두 같은 학습 데이터 기준 중앙값을 사용해 누수를 막는다.
  적용 기준과 건수는 아래 출력에 명시한다. (데이터 보호 2)
- 산점도: 테스트 데이터 실제값 + 학습으로 구한 회귀선

필요 라이브러리: pandas, numpy, matplotlib, scikit-learn
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
FIG_DIR = Path("outputs/figures")
TAB_DIR = Path("outputs/tables")


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # ---- 결측 처리: 학습 데이터 중앙값으로 대체 --------------------------
    median_distance = train_df["distance_km"].median()
    n_missing_train = int(train_df["distance_km"].isna().sum())
    n_missing_test = int(test_df["distance_km"].isna().sum())

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["distance_km_filled"] = train_df["distance_km"].fillna(median_distance)
    test_df["distance_km_filled"] = test_df["distance_km"].fillna(median_distance)

    print("=" * 78)
    print("[1] 결측 처리")
    print(f"    학습 데이터 distance_km 중앙값 : {median_distance:.3f} km")
    print(f"    학습 데이터 결측 대체 건수     : {n_missing_train}건 / {len(train_df):,}행")
    print(f"    테스트 데이터 결측 대체 건수   : {n_missing_test}건 / {len(test_df):,}행 (학습 중앙값 적용)")

    # ---- 선형회귀 학습 ---------------------------------------------------
    X_train = train_df[["distance_km_filled"]].values
    y_train = train_df["delivery_days"].values
    X_test = test_df[["distance_km_filled"]].values
    y_test = test_df["delivery_days"].values

    model = LinearRegression()
    model.fit(X_train, y_train)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    effect_100km = slope * 100

    print()
    print("[2] 회귀식")
    print(f"    delivery_days = {intercept:.4f} + {slope:.6f} * distance_km")
    print(f"    거리 100km 증가 시 예측 배송기간 변화: {effect_100km:+.3f} 일")

    # ---- 테스트 평가 -------------------------------------------------------
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    r2 = r2_score(y_test, y_pred)

    print()
    print("[3] 테스트 데이터 평가 (n=%d)" % len(test_df))
    print(f"    MAE  : {mae:.3f} 일")
    print(f"    RMSE : {rmse:.3f} 일")
    print(f"    R^2  : {r2:.4f}")

    # ---- 산점도 + 회귀선 -----------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    x_line = np.linspace(test_df["distance_km_filled"].min(), test_df["distance_km_filled"].max(), 200)
    y_line = model.predict(x_line.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        test_df["distance_km_filled"], y_test, s=8, alpha=0.25, color="#4C72B0", label="테스트 실제값"
    )
    ax.plot(x_line, y_line, color="#C44E52", linewidth=2, label="회귀선(학습 기준)")
    ax.set_title(f"distance_km 단순선형회귀 (테스트 n={len(test_df):,}, R²={r2:.3f})")
    ax.set_xlabel("distance_km (km, 결측은 학습 중앙값으로 대체)")
    ax.set_ylabel("delivery_days (일)")
    ax.legend()
    fig.tight_layout()
    fig_path = FIG_DIR / "06_linreg_distance_scatter.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print()
    print(f"[4] 저장: {fig_path.as_posix()}")

    # ---- 결과 표 저장 -------------------------------------------------------
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TAB_DIR / "06_linreg_distance_summary.csv"
    pd.DataFrame(
        [
            {"항목": "학습 distance_km 중앙값(결측 대체값)", "값": round(median_distance, 3)},
            {"항목": "학습 결측 대체 건수", "값": n_missing_train},
            {"항목": "테스트 결측 대체 건수", "값": n_missing_test},
            {"항목": "절편(intercept)", "값": round(intercept, 4)},
            {"항목": "기울기(slope, per km)", "값": round(slope, 6)},
            {"항목": "100km 증가당 예측 변화(일)", "값": round(effect_100km, 3)},
            {"항목": "MAE", "값": round(mae, 3)},
            {"항목": "RMSE", "값": round(rmse, 3)},
            {"항목": "R2", "값": round(r2, 4)},
        ]
    ).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[5] 저장: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
