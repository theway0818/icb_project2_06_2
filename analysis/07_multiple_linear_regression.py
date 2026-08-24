# -*- coding: utf-8 -*-
"""
analysis/03_variable_roles.py 에서 '예측에 사용'으로 분류한 12개 변수로
다중 선형회귀를 수행하고, 기준 모델(학습 평균) / 단순회귀(distance_km만)와
테스트 성능을 비교한다.

사용 변수(12개)
  수치형(8) : distance_km, item_count, seller_count, total_price_brl,
              total_freight_brl, total_weight_kg, total_volume_l, purchase_hour
  범주형(4) : customer_state, seller_state, primary_category, purchase_weekday
  * purchase_hour(0~23)는 시간의 선형적 흐름으로 보고 수치형으로 사용.
  * order_id, review_score, order_purchase_timestamp는 제외(03_variable_roles.py 기준).

결측 처리 (학습 데이터 기준으로 계산 후 학습·테스트 동일 적용, 데이터 보호 2)
  - 수치형: 학습 데이터 중앙값(median)
  - 범주형: 학습 데이터 최빈값(mode)

인코딩: 범주형 변수 원-핫 인코딩(OneHotEncoder, drop='first', 테스트에만 있는
        범주는 handle_unknown='ignore'로 전부 0 처리)

입력: data/processed/train.csv, test.csv (random_state=42, 80/20 분할)
필요 라이브러리: pandas, numpy, matplotlib(미사용 시 생략 가능), scikit-learn
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.stdout.reconfigure(encoding="utf-8")

TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
TAB_DIR = Path("outputs/tables")

NUMERIC_COLS = [
    "distance_km", "item_count", "seller_count", "total_price_brl",
    "total_freight_brl", "total_weight_kg", "total_volume_l", "purchase_hour",
]
CATEGORICAL_COLS = ["customer_state", "seller_state", "primary_category", "purchase_weekday"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
TARGET_COL = "delivery_days"


def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    y_train = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    # ---- 결측 현황(학습 기준 대체 전 원본 결측 건수) ------------------------
    print("=" * 78)
    print("[1] 결측 현황 및 대체 기준")
    for col in NUMERIC_COLS:
        n_tr, n_te = int(train_df[col].isna().sum()), int(test_df[col].isna().sum())
        if n_tr or n_te:
            print(f"    (수치/중앙값) {col:<18} 학습 결측 {n_tr}건, 테스트 결측 {n_te}건")
    for col in CATEGORICAL_COLS:
        n_tr, n_te = int(train_df[col].isna().sum()), int(test_df[col].isna().sum())
        if n_tr or n_te:
            print(f"    (범주/최빈값) {col:<18} 학습 결측 {n_tr}건, 테스트 결측 {n_te}건")

    # ---- 전처리 + 회귀 파이프라인 -------------------------------------------
    numeric_pipe = Pipeline([("impute_median", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        [
            ("impute_mode", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC_COLS),
            ("cat", categorical_pipe, CATEGORICAL_COLS),
        ]
    )
    model = Pipeline([("prep", preprocess), ("linreg", LinearRegression())])

    X_train = train_df[FEATURE_COLS]
    X_test = test_df[FEATURE_COLS]
    model.fit(X_train, y_train)

    n_dummy = model.named_steps["prep"].transform(X_train.iloc[:1]).shape[1]
    print()
    print(f"[2] 전처리 후 입력 열 수(원-핫 인코딩 포함): {n_dummy}개")
    print(f"    (수치형 {len(NUMERIC_COLS)}개 + 범주형 원-핫 더미 {n_dummy - len(NUMERIC_COLS)}개)")

    # ---- 세 모델 성능 비교 ---------------------------------------------------
    baseline_pred_value = y_train.mean()
    baseline_pred = np.full_like(y_test, fill_value=baseline_pred_value, dtype=float)
    mae_base, rmse_base, r2_base = eval_metrics(y_test, baseline_pred)

    simple_median = train_df["distance_km"].median()
    x_tr_simple = train_df["distance_km"].fillna(simple_median).values.reshape(-1, 1)
    x_te_simple = test_df["distance_km"].fillna(simple_median).values.reshape(-1, 1)
    simple_model = LinearRegression().fit(x_tr_simple, y_train)
    simple_pred = simple_model.predict(x_te_simple)
    mae_simple, rmse_simple, r2_simple = eval_metrics(y_test, simple_pred)

    multi_pred = model.predict(X_test)
    mae_multi, rmse_multi, r2_multi = eval_metrics(y_test, multi_pred)

    comparison = pd.DataFrame(
        [
            {"모델": "기준 모델(학습 평균)", "사용 변수 수": 0, "MAE": mae_base, "RMSE": rmse_base, "R2": r2_base},
            {"모델": "단순회귀(distance_km)", "사용 변수 수": 1, "MAE": mae_simple, "RMSE": rmse_simple, "R2": r2_simple},
            {"모델": "다중회귀(12개 변수)", "사용 변수 수": 12, "MAE": mae_multi, "RMSE": rmse_multi, "R2": r2_multi},
        ]
    )
    comparison[["MAE", "RMSE"]] = comparison[["MAE", "RMSE"]].round(3)
    comparison["R2"] = comparison["R2"].round(4)

    print()
    print("[3] 테스트 데이터 성능 비교 (n=%d)" % len(test_df))
    print(comparison.to_string(index=False))

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TAB_DIR / "07_model_comparison.csv"
    comparison.to_csv(out_path, index=False, encoding="utf-8-sig")
    print()
    print(f"[4] 저장: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
