# -*- coding: utf-8 -*-
"""
analysis/07_multiple_linear_regression.py 의 12변수 다중회귀에
아래 두 가지 특성 가공을 추가해 성능을 비교한다.

  1) distance_km 로그 전환: log1p(distance_km) = log(1 + distance_km)
     - 결측은 기존과 동일하게 학습 데이터 중앙값으로 먼저 대체한 뒤 로그 전환
       (원본 km 단위 결측 대체 기준을 유지, distance_km==0 건도 log1p(0)=0으로 안전 처리)
  2) 희소 지역 묶기: customer_state, seller_state 각각에서 학습 데이터 내
     비중이 1% 미만인 주(州)를 "OTHER"로 묶는다. 기준(1%)과 대상 목록,
     건수는 아래 출력에 남긴다. (데이터 보호 2)

세 모델(기준 / 단순회귀 / 12변수 다중회귀)에 이번 모델(로그+지역묶음 적용
다중회귀)을 더해 테스트 성능을 한 표로 비교한다.

입력: data/processed/train.csv, test.csv (random_state=42, 80/20 분할)
필요 라이브러리: pandas, numpy, scikit-learn
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

NUMERIC_COLS_RAW = [
    "distance_km", "item_count", "seller_count", "total_price_brl",
    "total_freight_brl", "total_weight_kg", "total_volume_l", "purchase_hour",
]
CATEGORICAL_COLS = ["customer_state", "seller_state", "primary_category", "purchase_weekday"]
FEATURE_COLS = NUMERIC_COLS_RAW + CATEGORICAL_COLS
TARGET_COL = "delivery_days"
RARE_THRESHOLD_RATIO = 0.01  # 학습 데이터 내 1% 미만인 州는 OTHER로 묶음
REGION_COLS = ["customer_state", "seller_state"]


def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH).copy()
    test_df = pd.read_csv(TEST_PATH).copy()
    n_train = len(train_df)

    y_train = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    # ---- 1) distance_km 로그 전환 -----------------------------------------
    dist_median = train_df["distance_km"].median()
    n_miss_tr = int(train_df["distance_km"].isna().sum())
    n_miss_te = int(test_df["distance_km"].isna().sum())
    train_df["distance_km_filled"] = train_df["distance_km"].fillna(dist_median)
    test_df["distance_km_filled"] = test_df["distance_km"].fillna(dist_median)
    train_df["distance_km_log"] = np.log1p(train_df["distance_km_filled"])
    test_df["distance_km_log"] = np.log1p(test_df["distance_km_filled"])

    print("=" * 78)
    print("[1] distance_km 로그 전환 (log1p = log(1+거리), 결측은 학습 중앙값 선대체)")
    print(f"    학습 중앙값(결측 대체값): {dist_median:.3f} km")
    print(f"    결측 대체 건수: 학습 {n_miss_tr}건, 테스트 {n_miss_te}건")
    print(f"    변환 전 범위: {train_df['distance_km_filled'].min():.3f} ~ {train_df['distance_km_filled'].max():.3f} km")
    print(f"    변환 후 범위: {train_df['distance_km_log'].min():.3f} ~ {train_df['distance_km_log'].max():.3f}")

    # ---- 2) 희소 지역(州) 묶기: 학습 데이터 비중 1% 미만 -> OTHER ------------
    print()
    print("[2] 희소 지역 묶기 (학습 데이터 기준 비중 1% 미만 -> OTHER)")
    for col in REGION_COLS:
        vc = train_df[col].value_counts()
        threshold = n_train * RARE_THRESHOLD_RATIO
        keep = set(vc[vc >= threshold].index)
        rare = vc[vc < threshold]
        print(f"    {col}: 전체 {vc.shape[0]}종 -> 유지 {len(keep)}종 + OTHER")
        rare_list = ", ".join(rare.index)
        print(f"        OTHER로 묶인 범주({len(rare)}종, 학습 {int(rare.sum())}행): {rare_list}")

        train_df[col + "_grp"] = train_df[col].where(train_df[col].isin(keep), other="OTHER")
        test_df[col + "_grp"] = test_df[col].where(test_df[col].isin(keep), other="OTHER")
        n_te_other = int((test_df[col + "_grp"] == "OTHER").sum())
        print(f"        테스트에서 OTHER로 분류된 행: {n_te_other}건 (학습 미등장 범주 포함)")

    numeric_cols_new = [c for c in NUMERIC_COLS_RAW if c != "distance_km"] + ["distance_km_log"]
    categorical_cols_new = ["customer_state_grp", "seller_state_grp", "primary_category", "purchase_weekday"]

    # ---- 파이프라인 구성 및 학습 -------------------------------------------
    numeric_pipe = Pipeline([("impute_median", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        [
            ("impute_mode", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_cols_new),
            ("cat", categorical_pipe, categorical_cols_new),
        ]
    )
    model = Pipeline([("prep", preprocess), ("linreg", LinearRegression())])

    X_train = train_df[numeric_cols_new + categorical_cols_new]
    X_test = test_df[numeric_cols_new + categorical_cols_new]
    model.fit(X_train, y_train)

    n_dummy = model.named_steps["prep"].transform(X_train.iloc[:1]).shape[1]
    print()
    print(f"[3] 전처리 후 입력 열 수: {n_dummy}개 (지역 묶기 전 07번 모델은 126개였음)")

    log_coef = float(model.named_steps["linreg"].coef_[numeric_cols_new.index("distance_km_log")])
    double_effect = log_coef * np.log(2)
    print(f"    distance_km_log 계수: {log_coef:.4f}")
    print(f"    -> 거리가 2배로 늘어날 때 예측 배송기간 변화: {double_effect:+.3f} 일 (로그 특성상 배수 단위로 해석)")

    # ---- 성능 비교 (기준 / 단순회귀 / 12변수 다중회귀 / 로그+지역묶음) --------
    baseline_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
    mae_base, rmse_base, r2_base = eval_metrics(y_test, baseline_pred)

    x_tr_simple = train_df["distance_km_filled"].values.reshape(-1, 1)
    x_te_simple = test_df["distance_km_filled"].values.reshape(-1, 1)
    simple_model = LinearRegression().fit(x_tr_simple, y_train)
    mae_simple, rmse_simple, r2_simple = eval_metrics(y_test, simple_model.predict(x_te_simple))

    numeric_cols_12 = list(NUMERIC_COLS_RAW)
    categorical_cols_12 = CATEGORICAL_COLS
    prep_12 = ColumnTransformer(
        [
            ("num", Pipeline([("impute_median", SimpleImputer(strategy="median"))]), numeric_cols_12),
            (
                "cat",
                Pipeline(
                    [
                        ("impute_mode", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
                    ]
                ),
                categorical_cols_12,
            ),
        ]
    )
    model_12 = Pipeline([("prep", prep_12), ("linreg", LinearRegression())])
    model_12.fit(train_df[numeric_cols_12 + categorical_cols_12], y_train)
    mae_12, rmse_12, r2_12 = eval_metrics(
        y_test, model_12.predict(test_df[numeric_cols_12 + categorical_cols_12])
    )

    multi_pred_new = model.predict(X_test)
    mae_new, rmse_new, r2_new = eval_metrics(y_test, multi_pred_new)

    comparison = pd.DataFrame(
        [
            {"모델": "기준 모델(학습 평균)", "사용 변수 수": 0, "MAE": mae_base, "RMSE": rmse_base, "R2": r2_base},
            {"모델": "단순회귀(distance_km)", "사용 변수 수": 1, "MAE": mae_simple, "RMSE": rmse_simple, "R2": r2_simple},
            {"모델": "다중회귀(12개 변수, 원본)", "사용 변수 수": 12, "MAE": mae_12, "RMSE": rmse_12, "R2": r2_12},
            {"모델": "다중회귀(거리 로그+지역묶음)", "사용 변수 수": 12, "MAE": mae_new, "RMSE": rmse_new, "R2": r2_new},
        ]
    )
    comparison[["MAE", "RMSE"]] = comparison[["MAE", "RMSE"]].round(3)
    comparison["R2"] = comparison["R2"].round(4)

    print()
    print("[4] 테스트 데이터 성능 비교 (n=%d)" % len(test_df))
    print(comparison.to_string(index=False))

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TAB_DIR / "08_model_comparison_log_grouped.csv"
    comparison.to_csv(out_path, index=False, encoding="utf-8-sig")
    print()
    print(f"[5] 저장: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
