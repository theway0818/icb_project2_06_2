# -*- coding: utf-8 -*-
"""
analysis/08_log_distance_grouped_regions.py 를 확장해, customer_state의
희소 범주를 무의미한 "OTHER"로 묶는 대신 브라질의 공식 5대 권역
(북부/북동부/중서부/남동부/남부)으로 재분류한다.

- STATE_TO_REGION 매핑은 브라질의 표준 행정구역 분류(외부 참조 지식)이며
  원본 데이터에서 만들어낸 값이 아니다. (CLAUDE.md 원칙 8과 무관한 별도 참조표)
- customer_state 27종 전체를 5개 권역으로 교체(희소 주만이 아니라 전체 적용).
- seller_state는 08번과 동일하게 학습 데이터 비중 1% 미만 -> OTHER 그룹핑 유지.
- distance_km는 08번과 동일하게 결측을 학습 중앙값으로 대체한 뒤 log1p 전환.

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
TARGET_COL = "delivery_days"
RARE_THRESHOLD_RATIO = 0.01  # seller_state: 학습 데이터 내 1% 미만 -> OTHER

# 브라질 공식 5대 권역 매핑 (외부 참조 지식, 원본 데이터 값이 아님)
STATE_TO_REGION = {
    "AC": "북부", "AP": "북부", "AM": "북부", "PA": "북부", "RO": "북부", "RR": "북부", "TO": "북부",
    "AL": "북동부", "BA": "북동부", "CE": "북동부", "MA": "북동부", "PB": "북동부",
    "PE": "북동부", "PI": "북동부", "RN": "북동부", "SE": "북동부",
    "DF": "중서부", "GO": "중서부", "MS": "중서부", "MT": "중서부",
    "ES": "남동부", "MG": "남동부", "RJ": "남동부", "SP": "남동부",
    "PR": "남부", "RS": "남부", "SC": "남부",
}


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

    # ---- 0) customer_state -> 브라질 5대 권역 매핑 검증 및 적용 --------------
    print("=" * 78)
    print("[0] customer_state -> 브라질 5대 권역 매핑")
    unmapped_train = sorted(set(train_df["customer_state"].dropna().unique()) - set(STATE_TO_REGION))
    unmapped_test = sorted(set(test_df["customer_state"].dropna().unique()) - set(STATE_TO_REGION))
    if unmapped_train or unmapped_test:
        raise ValueError(
            f"STATE_TO_REGION에 없는 customer_state 값 발견: 학습={unmapped_train}, 테스트={unmapped_test}"
        )

    train_df["customer_region"] = train_df["customer_state"].map(STATE_TO_REGION)
    test_df["customer_region"] = test_df["customer_state"].map(STATE_TO_REGION)

    region_counts_train = train_df["customer_region"].value_counts()
    region_counts_test = test_df["customer_region"].value_counts()
    print(f"    {'권역':<8}{'학습 건수':>10}{'테스트 건수':>12}")
    for region in ["북부", "북동부", "중서부", "남동부", "남부"]:
        print(
            f"    {region:<8}{int(region_counts_train.get(region, 0)):>10,}"
            f"{int(region_counts_test.get(region, 0)):>12,}"
        )
    print(f"    합계: 학습 {region_counts_train.sum():,} / 테스트 {region_counts_test.sum():,}")
    print("    (참고: RO/AM/AC/RR/AP 등 희소 주는 모두 '북부'로 매핑됨)")

    # ---- 1) distance_km 로그 전환 (08과 동일) -----------------------------
    dist_median = train_df["distance_km"].median()
    n_miss_tr = int(train_df["distance_km"].isna().sum())
    n_miss_te = int(test_df["distance_km"].isna().sum())
    train_df["distance_km_filled"] = train_df["distance_km"].fillna(dist_median)
    test_df["distance_km_filled"] = test_df["distance_km"].fillna(dist_median)
    train_df["distance_km_log"] = np.log1p(train_df["distance_km_filled"])
    test_df["distance_km_log"] = np.log1p(test_df["distance_km_filled"])

    print()
    print("[1] distance_km 로그 전환 (log1p, 결측은 학습 중앙값 선대체)")
    print(f"    학습 중앙값(결측 대체값): {dist_median:.3f} km")
    print(f"    결측 대체 건수: 학습 {n_miss_tr}건, 테스트 {n_miss_te}건")

    # ---- 2) seller_state 희소 그룹핑 (08과 동일: 1% 미만 -> OTHER) ---------
    print()
    print("[2] seller_state 희소 그룹핑 (학습 데이터 기준 비중 1% 미만 -> OTHER)")
    vc = train_df["seller_state"].value_counts()
    threshold = n_train * RARE_THRESHOLD_RATIO
    keep = set(vc[vc >= threshold].index)
    rare = vc[vc < threshold]
    print(f"    seller_state: 전체 {vc.shape[0]}종 -> 유지 {len(keep)}종 + OTHER")
    print(f"        OTHER로 묶인 범주({len(rare)}종, 학습 {int(rare.sum())}행): {', '.join(rare.index)}")
    train_df["seller_state_grp"] = train_df["seller_state"].where(train_df["seller_state"].isin(keep), other="OTHER")
    test_df["seller_state_grp"] = test_df["seller_state"].where(test_df["seller_state"].isin(keep), other="OTHER")

    numeric_cols_new = [c for c in NUMERIC_COLS_RAW if c != "distance_km"] + ["distance_km_log"]
    categorical_cols_new = ["customer_region", "seller_state_grp", "primary_category", "purchase_weekday"]

    # ---- 파이프라인 구성 및 학습 -------------------------------------------
    def build_pipeline(numeric_cols, categorical_cols):
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
        return Pipeline([("prep", preprocess), ("linreg", LinearRegression())])

    model = build_pipeline(numeric_cols_new, categorical_cols_new)
    X_train = train_df[numeric_cols_new + categorical_cols_new]
    X_test = test_df[numeric_cols_new + categorical_cols_new]
    model.fit(X_train, y_train)

    n_dummy = model.named_steps["prep"].transform(X_train.iloc[:1]).shape[1]
    print()
    print(f"[3] 전처리 후 입력 열 수: {n_dummy}개 (08번의 지역묶음 모델은 101개였음)")

    log_coef = float(model.named_steps["linreg"].coef_[numeric_cols_new.index("distance_km_log")])
    print(f"    distance_km_log 계수: {log_coef:.4f} (거리 2배 증가 시 {log_coef * np.log(2):+.3f}일)")

    # ---- 성능 비교: 기준 / 단순회귀 / 12변수 원본 / 08(로그+지역OTHER) / 09(권역) --
    baseline_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
    mae_base, rmse_base, r2_base = eval_metrics(y_test, baseline_pred)

    x_tr_simple = train_df["distance_km_filled"].values.reshape(-1, 1)
    x_te_simple = test_df["distance_km_filled"].values.reshape(-1, 1)
    simple_model = LinearRegression().fit(x_tr_simple, y_train)
    mae_simple, rmse_simple, r2_simple = eval_metrics(y_test, simple_model.predict(x_te_simple))

    model_12 = build_pipeline(NUMERIC_COLS_RAW, CATEGORICAL_COLS)
    model_12.fit(train_df[NUMERIC_COLS_RAW + CATEGORICAL_COLS], y_train)
    mae_12, rmse_12, r2_12 = eval_metrics(
        y_test, model_12.predict(test_df[NUMERIC_COLS_RAW + CATEGORICAL_COLS])
    )

    numeric_cols_08 = [c for c in NUMERIC_COLS_RAW if c != "distance_km"] + ["distance_km_log"]
    for col, keep_col in [("customer_state", "customer_state_grp08"), ("seller_state", "seller_state_grp")]:
        vc8 = train_df[col].value_counts()
        keep8 = set(vc8[vc8 >= threshold].index)
        train_df[keep_col] = train_df[col].where(train_df[col].isin(keep8), other="OTHER")
        test_df[keep_col] = test_df[col].where(test_df[col].isin(keep8), other="OTHER")
    categorical_cols_08 = ["customer_state_grp08", "seller_state_grp", "primary_category", "purchase_weekday"]
    model_08 = build_pipeline(numeric_cols_08, categorical_cols_08)
    model_08.fit(train_df[numeric_cols_08 + categorical_cols_08], y_train)
    mae_08, rmse_08, r2_08 = eval_metrics(
        y_test, model_08.predict(test_df[numeric_cols_08 + categorical_cols_08])
    )

    multi_pred_new = model.predict(X_test)
    mae_new, rmse_new, r2_new = eval_metrics(y_test, multi_pred_new)

    comparison = pd.DataFrame(
        [
            {"모델": "기준 모델(학습 평균)", "사용 변수 수": 0, "MAE": mae_base, "RMSE": rmse_base, "R2": r2_base},
            {"모델": "단순회귀(distance_km)", "사용 변수 수": 1, "MAE": mae_simple, "RMSE": rmse_simple, "R2": r2_simple},
            {"모델": "다중회귀(12개 변수, 원본)", "사용 변수 수": 12, "MAE": mae_12, "RMSE": rmse_12, "R2": r2_12},
            {"모델": "다중회귀(거리 로그+지역 OTHER묶음)", "사용 변수 수": 12, "MAE": mae_08, "RMSE": rmse_08, "R2": r2_08},
            {"모델": "다중회귀(거리 로그+customer 5대권역)", "사용 변수 수": 12, "MAE": mae_new, "RMSE": rmse_new, "R2": r2_new},
        ]
    )
    comparison[["MAE", "RMSE"]] = comparison[["MAE", "RMSE"]].round(3)
    comparison["R2"] = comparison["R2"].round(4)

    print()
    print("[4] 테스트 데이터 성능 비교 (n=%d)" % len(test_df))
    print(comparison.to_string(index=False))

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TAB_DIR / "09_model_comparison_customer_region.csv"
    comparison.to_csv(out_path, index=False, encoding="utf-8-sig")
    print()
    print(f"[5] 저장: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
