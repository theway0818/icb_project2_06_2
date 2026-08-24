# -*- coding: utf-8 -*-
"""
배송기간 예측 주문 페이지 (Streamlit).

최종 모델: "로그 전환만 적용한" 다중선형회귀
  - distance_km -> log1p(distance_km) 로그 전환 (analysis/06, 08과 동일 방식)
  - customer_state / seller_state는 희소 지역 묶기(08, 09번)를 적용하지 않고
    원본 범주 그대로 원-핫 인코딩 (analysis/03_variable_roles.py 기준 12개 변수)
  - 수치형 결측: 학습 데이터 중앙값 / 범주형 결측: 학습 데이터 최빈값 (07, 08과 동일)

화면 구성 (탭 2개)
  1) 주문 페이지: 주문 조건(12개 변수) 입력 폼
     - "예상 배송일 계산" 시 점 예측값(delivery_days) 표시
     - 테스트 데이터 잔차(실제-예측)의 10~90 백분위수를 이용한 80% 예측 구간을
       "더 빨리 도착/지연 가능 범위"로 함께 표시 (원본에 없는 값을 임의로 만들지
       않기 위해 실제 테스트 잔차 분포에서 구간을 계산함)
  2) 변수 영향 분석: 회귀계수를 시각화해 각 변수가 예측 배송기간에 미치는 영향을 보여줌
     - 수치형 변수: "1표준편차 증가 시 예측 배송일 변화(일)" = 계수 × 학습 데이터 표준편차
       (원 단위가 서로 다른 변수를 비교 가능하게 만들기 위한 표준 방식)
     - 범주형 변수: 각 범주의 회귀계수 = 기준 범주 대비 예측 배송일 차이(일)
       (기준 범주는 OneHotEncoder(drop='first')가 제외한 첫 범주)

입력: data/processed/train.csv, test.csv (analysis/04_train_test_split.py 산출물)
필요 라이브러리: streamlit, pandas, numpy, scikit-learn, plotly

실행: streamlit run analysis/11_streamlit_app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")

NUMERIC_COLS_RAW = [
    "distance_km", "item_count", "seller_count", "total_price_brl",
    "total_freight_brl", "total_weight_kg", "total_volume_l", "purchase_hour",
]
CATEGORICAL_COLS = ["customer_state", "seller_state", "primary_category", "purchase_weekday"]
TARGET_COL = "delivery_days"
WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@st.cache_data
def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


@st.cache_resource
def fit_model(train_df: pd.DataFrame):
    df = train_df.copy()
    dist_median = df["distance_km"].median()
    df["distance_km_filled"] = df["distance_km"].fillna(dist_median)
    df["distance_km_log"] = np.log1p(df["distance_km_filled"])

    numeric_cols = [c for c in NUMERIC_COLS_RAW if c != "distance_km"] + ["distance_km_log"]
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
                CATEGORICAL_COLS,
            ),
        ]
    )
    model = Pipeline([("prep", preprocess), ("linreg", LinearRegression())])
    X_train = df[numeric_cols + CATEGORICAL_COLS]
    y_train = df[TARGET_COL].values
    model.fit(X_train, y_train)
    return model, dist_median, numeric_cols


@st.cache_resource
def evaluate_model(_model, dist_median: float, numeric_cols: list, test_df: pd.DataFrame):
    df = test_df.copy()
    df["distance_km_filled"] = df["distance_km"].fillna(dist_median)
    df["distance_km_log"] = np.log1p(df["distance_km_filled"])
    X_test = df[numeric_cols + CATEGORICAL_COLS]
    y_test = df[TARGET_COL].values
    y_pred = _model.predict(X_test)
    residuals = y_test - y_pred  # 실제 - 예측 (양수: 실제가 더 늦음 / 음수: 실제가 더 빠름)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    r2 = r2_score(y_test, y_pred)
    q10, q90 = np.percentile(residuals, [10, 90])
    return {"mae": mae, "rmse": rmse, "r2": r2, "q10": q10, "q90": q90, "n_test": len(df)}


@st.cache_data
def compute_feature_effects(_model: Pipeline, train_df: pd.DataFrame, dist_median: float, numeric_cols: list):
    """회귀계수를 '예측 배송일에 대한 영향(일)'으로 환산한다.

    수치형: 계수 × 학습 데이터에서의 표준편차 (1표준편차 증가 시 변화량, 단위: 일)
    범주형: OneHotEncoder(drop='first') 계수 그대로 (기준 범주 대비 차이, 단위: 일)
    """
    df = train_df.copy()
    df["distance_km_filled"] = df["distance_km"].fillna(dist_median)
    df["distance_km_log"] = np.log1p(df["distance_km_filled"])

    prep: ColumnTransformer = _model.named_steps["prep"]
    linreg: LinearRegression = _model.named_steps["linreg"]
    coefs = linreg.coef_
    n_num = len(numeric_cols)

    num_pipeline = prep.named_transformers_["num"]
    X_num = num_pipeline.transform(df[numeric_cols])
    stds = np.std(X_num, axis=0, ddof=1)
    numeric_effect = pd.DataFrame(
        {"변수": numeric_cols, "효과": coefs[:n_num] * stds}
    ).sort_values("효과", key=lambda s: s.abs(), ascending=True)

    onehot: OneHotEncoder = prep.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = onehot.get_feature_names_out(CATEGORICAL_COLS)
    cat_coefs = coefs[n_num:]
    reference = {col: cats[0] for col, cats in zip(CATEGORICAL_COLS, onehot.categories_)}

    rows = []
    for name, coef in zip(cat_feature_names, cat_coefs):
        for col in sorted(CATEGORICAL_COLS, key=len, reverse=True):
            prefix = col + "_"
            if name.startswith(prefix):
                rows.append({"변수그룹": col, "범주": name[len(prefix):], "효과": coef})
                break
    categorical_effect = pd.DataFrame(rows)

    return numeric_effect, categorical_effect, reference


def render_order_tab(train_df, model, dist_median, numeric_cols, metrics) -> None:
    st.caption("주문 조건을 입력하면 예상 배송기간과 지연/단축 가능 범위를 보여줍니다.")

    customer_states = sorted(train_df["customer_state"].dropna().unique().tolist())
    seller_states = sorted(train_df["seller_state"].dropna().unique().tolist())
    categories = sorted(train_df["primary_category"].dropna().unique().tolist())

    with st.form("order_form"):
        st.subheader("배송 지역")
        col1, col2 = st.columns(2)
        with col1:
            customer_state = st.selectbox("고객 지역(customer_state)", customer_states, index=customer_states.index("SP"))
        with col2:
            seller_state = st.selectbox("판매자 지역(seller_state)", seller_states, index=seller_states.index("SP"))
        distance_km = st.number_input(
            "예상 배송 거리(distance_km, km)", min_value=0.0, max_value=4000.0, value=427.4, step=10.0
        )

        st.subheader("상품 정보")
        col3, col4 = st.columns(2)
        with col3:
            primary_category = st.selectbox("대표 상품 카테고리(primary_category)", categories, index=categories.index("bed_bath_table"))
            item_count = st.number_input("상품 수량(item_count)", min_value=1, max_value=20, value=1, step=1)
            seller_count = st.number_input("판매자 수(seller_count)", min_value=1, max_value=5, value=1, step=1)
        with col4:
            total_price_brl = st.number_input("상품 금액 합계(total_price_brl, BRL)", min_value=0.0, value=86.0, step=10.0)
            total_freight_brl = st.number_input("배송비 합계(total_freight_brl, BRL)", min_value=0.0, value=17.0, step=5.0)

        col5, col6 = st.columns(2)
        with col5:
            total_weight_kg = st.number_input("총 중량(total_weight_kg, kg)", min_value=0.0, value=0.75, step=0.5)
        with col6:
            total_volume_l = st.number_input("총 부피(total_volume_l, L)", min_value=0.0, value=7.4, step=1.0)

        st.subheader("주문 시점")
        col7, col8 = st.columns(2)
        with col7:
            purchase_weekday = st.selectbox("주문 요일(purchase_weekday)", WEEKDAY_ORDER, index=0)
        with col8:
            purchase_hour = st.slider("주문 시각(purchase_hour, 시)", min_value=0, max_value=23, value=15)

        submitted = st.form_submit_button("예상 배송일 계산", use_container_width=True)

    if submitted:
        input_row = pd.DataFrame(
            [
                {
                    "item_count": item_count,
                    "seller_count": seller_count,
                    "total_price_brl": total_price_brl,
                    "total_freight_brl": total_freight_brl,
                    "total_weight_kg": total_weight_kg,
                    "total_volume_l": total_volume_l,
                    "purchase_hour": purchase_hour,
                    "distance_km_log": np.log1p(distance_km),
                    "customer_state": customer_state,
                    "seller_state": seller_state,
                    "primary_category": primary_category,
                    "purchase_weekday": purchase_weekday,
                }
            ]
        )
        pred = float(model.predict(input_row[numeric_cols + CATEGORICAL_COLS])[0])
        low = max(0.0, pred + metrics["q10"])
        high = pred + metrics["q90"]

        st.divider()
        st.subheader("예측 결과")
        st.metric("예상 배송기간", f"{pred:.1f} 일")
        st.info(
            f"비슷한 과거 주문(테스트 데이터 기준)의 80%가 실제로 도착한 범위: "
            f"**{low:.1f}일 ~ {high:.1f}일**\n\n"
            f"(더 빠르면 예상보다 약 {abs(min(metrics['q10'], 0)):.1f}일 단축, "
            f"더 늦으면 약 {max(metrics['q90'], 0):.1f}일 지연될 수 있습니다)"
        )
        st.caption(
            f"모델 성능(테스트 n={metrics['n_test']:,}): "
            f"MAE {metrics['mae']:.2f}일 · RMSE {metrics['rmse']:.2f}일 · R² {metrics['r2']:.3f}. "
            "이 값은 참고용 추정치이며 실제 배송일과 다를 수 있습니다."
        )

    with st.expander("모델 정보"):
        st.write(
            "- 사용 변수(12개): distance_km(로그 전환), item_count, seller_count, "
            "total_price_brl, total_freight_brl, total_weight_kg, total_volume_l, "
            "purchase_hour, customer_state, seller_state, primary_category, purchase_weekday"
        )
        st.write("- 제외 변수: order_id(식별자), review_score(주문 이후 생성 정보), order_purchase_timestamp(원본)")
        st.write(
            f"- 결측 처리: 수치형은 학습 데이터 중앙값, 범주형은 학습 데이터 최빈값으로 대체 "
            f"(distance_km 중앙값 대체값: {dist_median:.1f}km)"
        )
        st.write(f"- 테스트 MAE {metrics['mae']:.2f}일 / RMSE {metrics['rmse']:.2f}일 / R² {metrics['r2']:.3f}")


def render_effects_tab(train_df, model, dist_median, numeric_cols) -> None:
    st.caption(
        "회귀계수를 '예측 배송일에 미치는 영향(일)'으로 환산해 보여줍니다. "
        "막대가 오른쪽(양수, 빨강)이면 배송기간을 늘리는 방향, 왼쪽(음수, 파랑)이면 줄이는 방향입니다."
    )
    numeric_effect, categorical_effect, reference = compute_feature_effects(
        model, train_df, dist_median, numeric_cols
    )

    st.subheader("수치형 변수: 1표준편차 증가 시 예측 배송일 변화")
    fig_num = px.bar(
        numeric_effect,
        x="효과",
        y="변수",
        orientation="h",
        color=numeric_effect["효과"] > 0,
        color_discrete_map={True: "#C44E52", False: "#4C72B0"},
        labels={"효과": "예측 배송일 변화(일)", "변수": ""},
    )
    fig_num.update_layout(showlegend=False, height=380, margin=dict(l=10, r=10, t=10, b=10))
    fig_num.add_vline(x=0, line_color="gray", line_width=1)
    st.plotly_chart(fig_num, use_container_width=True)
    st.caption(
        "예: distance_km_log가 +X일이면, 거리가 표준편차만큼 늘어날 때 평균적으로 배송이 X일 더 걸린다는 뜻입니다. "
        "서로 단위(km, kg, BRL 등)가 달라 원래 계수는 직접 비교할 수 없어, '1표준편차 변화당 효과'로 환산해 비교 가능하게 만들었습니다."
    )

    st.divider()
    st.subheader("범주형 변수: 기준 범주 대비 예측 배송일 차이")
    group_labels = {
        "customer_state": "고객 지역(customer_state)",
        "seller_state": "판매자 지역(seller_state)",
        "primary_category": "상품 카테고리(primary_category)",
        "purchase_weekday": "주문 요일(purchase_weekday)",
    }
    selected_label = st.selectbox("변수 선택", list(group_labels.values()))
    selected_col = [k for k, v in group_labels.items() if v == selected_label][0]

    sub = categorical_effect[categorical_effect["변수그룹"] == selected_col].sort_values("효과", ascending=True)
    n_show = len(sub)
    st.caption(f"기준 범주(효과 0으로 취급): **{reference[selected_col]}** · 전체 {n_show}개 범주 표시")

    fig_cat = px.bar(
        sub,
        x="효과",
        y="범주",
        orientation="h",
        color=sub["효과"] > 0,
        color_discrete_map={True: "#C44E52", False: "#4C72B0"},
        labels={"효과": "기준 범주 대비 예측 배송일 차이(일)", "범주": ""},
    )
    fig_cat.update_layout(
        showlegend=False, height=max(300, 22 * n_show), margin=dict(l=10, r=10, t=10, b=10)
    )
    fig_cat.add_vline(x=0, line_color="gray", line_width=1)
    st.plotly_chart(fig_cat, use_container_width=True)
    st.caption(
        f"예: 어떤 범주의 효과가 +3일이면, 다른 조건이 같을 때 기준 범주('{reference[selected_col]}')보다 "
        "그 범주에서 평균적으로 배송이 3일 더 걸렸다는 뜻입니다."
    )


def main() -> None:
    st.set_page_config(page_title="배송기간 예측 대시보드", page_icon="📦", layout="wide")
    st.title("📦 배송기간 예측 대시보드")

    train_df, test_df = load_data()
    model, dist_median, numeric_cols = fit_model(train_df)
    metrics = evaluate_model(model, dist_median, numeric_cols, test_df)

    tab_order, tab_effects = st.tabs(["🧾 주문 페이지", "📊 변수 영향 분석"])
    with tab_order:
        render_order_tab(train_df, model, dist_median, numeric_cols, metrics)
    with tab_effects:
        render_effects_tab(train_df, model, dist_median, numeric_cols)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
