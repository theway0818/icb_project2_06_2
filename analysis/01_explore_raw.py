# -*- coding: utf-8 -*-
"""
data/raw/olist_delivery_orders_sample.csv 원본 구조 확인 스크립트.

수행 내용
  1) 행 단위(관측 단위) 확인: order_id 중복 여부로 판정
  2) 열 목록 및 자료형/결측/고유값/값 범위 요약
  3) 열 의미 추정표(관측된 값 근거 기반, 추정임을 명시)
  4) 범주형 분포 및 값 특이사항 점검

필요 라이브러리: pandas, numpy (표준 라이브러리: sys, pathlib)
원본 파일은 읽기만 하며 수정하지 않는다.
"""

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

RAW_PATH = Path("data/raw/olist_delivery_orders_sample.csv")
OUT_DIR = Path("outputs/tables")

# 열 의미 추정(원본에 정의서가 없어 열 이름과 관측된 값을 근거로 추정한 내용)
COLUMN_MEANING = {
    "order_id": ("주문 식별자", "주문 1건을 구분하는 해시 문자열. 분석용 수치변수로 사용하지 않음"),
    "order_purchase_timestamp": ("주문 시각", "고객이 주문을 넣은 시점(초 단위 타임스탬프)"),
    "delivery_days": ("배송 소요일수", "주문 시점부터 고객 수령까지 걸린 일수(소수점=시간 단위 포함)"),
    "review_score": ("리뷰 평점", "고객 만족도 점수(1~5). 0.5 단위 값이 있어 주문 내 복수 리뷰의 평균으로 추정"),
    "customer_state": ("고객 주(州)", "브라질 주 약어로 표기된 배송지 지역"),
    "seller_state": ("판매자 주(州)", "브라질 주 약어로 표기된 발송지 지역"),
    "distance_km": ("배송 거리(km)", "판매자 지역과 고객 지역 사이의 추정 직선거리"),
    "item_count": ("주문 상품 수량", "해당 주문에 포함된 상품 아이템 개수"),
    "seller_count": ("판매자 수", "해당 주문에 관여한 서로 다른 판매자 수"),
    "total_price_brl": ("상품 금액 합계(BRL)", "주문 내 상품 가격 합계, 브라질 헤알"),
    "total_freight_brl": ("배송비 합계(BRL)", "주문 내 배송비 합계, 브라질 헤알"),
    "total_weight_kg": ("총 중량(kg)", "주문 상품의 무게 합계"),
    "total_volume_l": ("총 부피(L)", "주문 상품의 부피 합계(치수 환산 추정)"),
    "primary_category": ("대표 상품 카테고리", "주문의 주된 상품 카테고리명(영문 소문자)"),
    "purchase_weekday": ("주문 요일", "order_purchase_timestamp에서 파생된 영문 요일 약어"),
    "purchase_hour": ("주문 시각(시)", "order_purchase_timestamp에서 파생된 0~23 시"),
}

# 주문 이후에 확정되는 정보(예측 입력으로 쓰면 정보 누수)
POST_ORDER_COLS = ["delivery_days", "review_score"]


def sample_values(s: pd.Series, n: int = 3) -> str:
    return ", ".join(str(v) for v in s.dropna().unique()[:n])


def value_range(s: pd.Series) -> str:
    nn = s.dropna()
    if nn.empty:
        return "-"
    if pd.api.types.is_numeric_dtype(s):
        return f"{nn.min():g} ~ {nn.max():g}"
    if pd.api.types.is_datetime64_any_dtype(s):
        return f"{nn.min()} ~ {nn.max()}"
    return "-"


def column_group(col: str) -> str:
    if col == "order_id":
        return "식별자"
    if col in POST_ORDER_COLS:
        return "주문 이후 생성"
    if col in ("purchase_weekday", "purchase_hour"):
        return "파생(시간)"
    return "주문 시점 정보"


def main() -> None:
    df = pd.read_csv(RAW_PATH)
    n_rows, n_cols = df.shape

    print("=" * 78)
    print(f"[1] 파일: {RAW_PATH.as_posix()}")
    print(f"    행 수: {n_rows:,} / 열 수: {n_cols}")
    print("=" * 78)

    # ---- 행 단위 판정 ---------------------------------------------------
    n_unique_order = df["order_id"].nunique()
    n_dup_rows = int(df["order_id"].duplicated().sum())
    n_dup_ids = int((df["order_id"].value_counts() > 1).sum())
    n_dup_full = int(df.duplicated().sum())

    print()
    print("[2] 행 단위(관측 단위) 확인")
    print(f"    order_id 고유값 수   : {n_unique_order:,}")
    print(f"    order_id 중복 행 수  : {n_dup_rows:,} (중복된 order_id 종류: {n_dup_ids:,})")
    print(f"    전체 열 완전중복 행  : {n_dup_full:,}")
    if n_dup_rows == 0:
        print("    -> 판정: 1행 = 주문 1건(order_id 기준 유일). 주문 단위로 집계된 테이블.")
    else:
        print("    -> 판정: order_id 중복 존재. 주문보다 작은 단위이거나 중복 적재 가능성.")

    # ---- 변수 요약 ------------------------------------------------------
    ts = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    n_ts_bad = int(ts.isna().sum() - df["order_purchase_timestamp"].isna().sum())

    summary = pd.DataFrame(
        [
            {
                "열이름": col,
                "자료형": str(df[col].dtype),
                "결측수": int(df[col].isna().sum()),
                "결측률(%)": round(float(df[col].isna().mean()) * 100, 2),
                "고유값수": int(df[col].nunique(dropna=True)),
                "값범위": value_range(ts if col == "order_purchase_timestamp" else df[col]),
                "예시값": sample_values(df[col]),
            }
            for col in df.columns
        ]
    )

    print()
    print("[3] 변수 요약 (자료형 / 결측 / 고유값 / 값범위)")
    print(summary.to_string(index=False))
    print(f"    order_purchase_timestamp 파싱 실패 건수: {n_ts_bad:,}")

    # ---- 열 의미 추정표 --------------------------------------------------
    meaning = pd.DataFrame(
        [
            {
                "열이름": col,
                "추정 의미": COLUMN_MEANING.get(col, ("(미정의)", ""))[0],
                "설명(추정)": COLUMN_MEANING.get(col, ("", "(사전에 없는 열)"))[1],
                "구분": column_group(col),
            }
            for col in df.columns
        ]
    )
    print()
    print("[4] 열 의미 추정표 (원본에 정의서가 없어 열 이름과 값 분포로 추정)")
    print(meaning.to_string(index=False))

    # ---- 범주형 분포 -----------------------------------------------------
    print()
    print("[5] 주요 범주형 분포")
    for col in ["customer_state", "seller_state", "primary_category", "purchase_weekday", "review_score"]:
        vc = df[col].value_counts(dropna=False)
        top = ", ".join(f"{k}({v:,})" for k, v in vc.head(5).items())
        print(f"    {col:<18} 고유 {vc.shape[0]:>3}종(결측 포함) | 상위5: {top}")

    # ---- 값 특이사항 -----------------------------------------------------
    print()
    print("[6] 값 특이사항 점검")
    n_half = int((df["review_score"].dropna() % 1 != 0).sum())
    print(f"    review_score 정수가 아닌 값 : {n_half:,}건 (주문 내 복수 리뷰의 평균으로 추정)")
    print(f"    distance_km == 0            : {int((df['distance_km'] == 0).sum()):,}건 (동일 지역 추정)")
    print(f"    total_freight_brl == 0      : {int((df['total_freight_brl'] == 0).sum()):,}건 (무료배송 추정)")
    print(f"    결측이 하나 이상 있는 행    : {int(df.isna().any(axis=1).sum()):,}건 / {n_rows:,}행")

    # ---- 누수 점검 -------------------------------------------------------
    print()
    print("[7] 예측 입력 사용 시 확인 필요(주문 이후 확정되는 정보)")
    for col in POST_ORDER_COLS:
        print(f"    - {col}: 주문 시점에는 알 수 없는 값 -> 설명변수로 사용하지 않도록 확인 필요")

    # ---- 저장 ------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "01_raw_column_profile.csv"
    meaning.merge(summary, on="열이름", how="left").to_csv(out_path, index=False, encoding="utf-8-sig")
    print()
    print(f"[8] 저장: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
