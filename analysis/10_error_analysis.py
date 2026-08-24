# -*- coding: utf-8 -*-
"""
09번 다중회귀 모델(distance_km 로그 전환 + customer_state 5대권역 +
seller_state 희소그룹핑, 12개 변수)의 테스트 예측 오차를 분석한다.

- 절대오차(|실제-예측|)가 큰 주문 상위 20건 출력
- 고객 지역(5대권역)별 데이터 수·MAE 비교
- 거리 구간별 데이터 수·MAE 비교

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
from sklearn.metrics import mean_absolute_error
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
TARGET_COL = "delivery_days"
RARE_THRESHOLD_RATIO = 0.01

STATE_TO_REGION = {
    "AC": "북부", "AP": "북부", "AM": "북부", "PA": "북부", "RO": "북부", "RR": "북부", "TO": "북부",
    "AL": "북동부", "BA": "북동부", "CE": "북동부", "MA": "북동부", "PB": "북동부",
    "PE": "북동부", "PI": "북동부", "RN": "북동부", "SE": "북동부",
    "DF": "중서부", "GO": "중서부", "MS": "중서부", "MT": "중서부",
    "ES": "남동부", "MG": "남동부", "RJ": "남동부", "SP": "남동부",
    "PR": "남부", "RS": "남부", "SC": "남부",
}

DISTANCE_BINS = [0, 200, 500, 1000, 2000, np.inf]
DISTANCE_LABELS = ["0-200km", "200-500km", "500-1000km", "1000-2000km", "2000km+"]


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH).copy()
    test_df = pd.read_csv(TEST_PATH).copy()
    n_train = len(train_df)

    y_train = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    # ---- 09번과 동일한 전처리 재현 -----------------------------------------
    train_df["customer_region"] = train_df["customer_state"].map(STATE_TO_REGION)
    test_df["customer_region"] = test_df["customer_state"].map(STATE_TO_REGION)

    dist_median = train_df["distance_km"].median()
    train_df["distance_km_filled"] = train_df["distance_km"].fillna(dist_median)
    test_df["distance_km_filled"] = test_df["distance_km"].fillna(dist_median)
    train_df["distance_km_log"] = np.log1p(train_df["distance_km_filled"])
    test_df["distance_km_log"] = np.log1p(test_df["distance_km_filled"])

    vc = train_df["seller_state"].value_counts()
    threshold = n_train * RARE_THRESHOLD_RATIO
    keep = set(vc[vc >= threshold].index)
    train_df["seller_state_grp"] = train_df["seller_state"].where(train_df["seller_state"].isin(keep), other="OTHER")
    test_df["seller_state_grp"] = test_df["seller_state"].where(test_df["seller_state"].isin(keep), other="OTHER")

    numeric_cols = [c for c in NUMERIC_COLS_RAW if c != "distance_km"] + ["distance_km_log"]
    categorical_cols = ["customer_region", "seller_state_grp", "primary_category", "purchase_weekday"]

    preprocess = ColumnTransformer(
        [
            ("num", Pipeline([("impute_median", SimpleImputer(strategy="median"))]), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("impute_mode", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )
    model = Pipeline([("prep", preprocess), ("linreg", LinearRegression())])
    X_train = train_df[numeric_cols + categorical_cols]
    X_test = test_df[numeric_cols + categorical_cols]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    result = test_df[["order_id", "customer_state", "customer_region", "distance_km_filled"]].copy()
    result["실제_delivery_days"] = y_test
    result["예측_delivery_days"] = y_pred
    result["절대오차"] = np.abs(result["실제_delivery_days"] - result["예측_delivery_days"])
    result["거리구간"] = pd.cut(result["distance_km_filled"], bins=DISTANCE_BINS, labels=DISTANCE_LABELS, right=False)

    overall_mae = mean_absolute_error(y_test, y_pred)
    print("=" * 78)
    print(f"[0] 09번 다중회귀 모델 테스트 전체 MAE: {overall_mae:.3f}일 (n={len(test_df):,})")

    # ---- 절대오차 상위 20건 ---------------------------------------------
    top20 = result.sort_values("절대오차", ascending=False).head(20).reset_index(drop=True)
    top20_display = top20[
        ["order_id", "customer_state", "customer_region", "distance_km_filled",
         "실제_delivery_days", "예측_delivery_days", "절대오차"]
    ].round({"distance_km_filled": 1, "실제_delivery_days": 2, "예측_delivery_days": 2, "절대오차": 2})

    print()
    print("[1] 절대오차 상위 20건")
    print(top20_display.to_string(index=False))

    # ---- 고객 지역별 데이터 수 / MAE ---------------------------------------
    region_order = ["북부", "북동부", "중서부", "남동부", "남부"]
    by_region = (
        result.groupby("customer_region")["절대오차"]
        .agg(데이터수="count", MAE="mean")
        .reindex(region_order)
        .reset_index()
        .rename(columns={"customer_region": "고객지역"})
    )
    by_region["MAE"] = by_region["MAE"].round(3)

    print()
    print("[2] 고객 지역별 데이터 수 / MAE")
    print(by_region.to_string(index=False))

    # ---- 거리 구간별 데이터 수 / MAE ---------------------------------------
    by_distance = (
        result.groupby("거리구간", observed=True)["절대오차"]
        .agg(데이터수="count", MAE="mean")
        .reindex(DISTANCE_LABELS)
        .reset_index()
    )
    by_distance["MAE"] = by_distance["MAE"].round(3)

    print()
    print("[3] 거리 구간별 데이터 수 / MAE")
    print(by_distance.to_string(index=False))

    # ---- 저장 --------------------------------------------------------------
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    top20_display.to_csv(TAB_DIR / "10_top20_abs_error.csv", index=False, encoding="utf-8-sig")
    by_region.to_csv(TAB_DIR / "10_mae_by_customer_region.csv", index=False, encoding="utf-8-sig")
    by_distance.to_csv(TAB_DIR / "10_mae_by_distance_bin.csv", index=False, encoding="utf-8-sig")
    print()
    print("[4] 저장: outputs/tables/10_top20_abs_error.csv")
    print("    저장: outputs/tables/10_mae_by_customer_region.csv")
    print("    저장: outputs/tables/10_mae_by_distance_bin.csv")


if __name__ == "__main__":
    main()
