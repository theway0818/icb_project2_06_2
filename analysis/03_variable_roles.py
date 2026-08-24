# -*- coding: utf-8 -*-
"""
data/raw/olist_delivery_orders_sample.csv 변수 역할 분류표 생성.

목표 변수: delivery_days (배송 소요일수 예측)
분류 기준
  - 식별자      : 분석용 수치변수로 쓰지 않는 행 구분자 (데이터 보호 3)
  - 목표 변수   : 예측 대상
  - 예측에 사용 : 주문 시점에 이미 알 수 있는 정보만 포함
  - 예측에서 제외: order_id, review_score(요청에 따른 제외 + 주문 이후 생성 정보이므로
                   데이터 보호 4 위반), order_purchase_timestamp(원본 그대로는 미사용,
                   purchase_weekday/purchase_hour로 파생하여 대신 사용)

필요 라이브러리: pandas (표준 라이브러리: sys, pathlib)
원본 파일은 열 이름 확인 용도로만 읽으며 수정하지 않는다.
"""

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

RAW_PATH = Path("data/raw/olist_delivery_orders_sample.csv")
OUT_DIR = Path("outputs/tables")

ROLE_MAP = {
    "order_id": ("식별자", "주문 식별용 해시값. 수치적 의미 없음 (데이터 보호 3)"),
    "order_purchase_timestamp": ("예측에서 제외", "원본 형태로는 미사용. purchase_weekday/purchase_hour로 파생하여 대신 사용"),
    "delivery_days": ("목표 변수", "예측 대상(배송 소요일수)"),
    "review_score": ("예측에서 제외", "요청에 따라 제외 + 배송 완료 후에 남는 정보라 예측 시점에 알 수 없음(데이터 보호 4)"),
    "customer_state": ("예측에 사용", "주문 시점에 이미 확정된 배송지 지역"),
    "seller_state": ("예측에 사용", "주문 시점에 이미 확정된 발송지 지역"),
    "distance_km": ("예측에 사용", "고객/판매자 지역으로 주문 시점에 계산 가능한 거리"),
    "item_count": ("예측에 사용", "주문 시점에 확정되는 상품 수량"),
    "seller_count": ("예측에 사용", "주문 시점에 확정되는 판매자 수"),
    "total_price_brl": ("예측에 사용", "주문 시점에 확정되는 상품 금액"),
    "total_freight_brl": ("예측에 사용", "주문 시점에 확정되는 배송비"),
    "total_weight_kg": ("예측에 사용", "주문 시점에 확정되는 상품 총중량"),
    "total_volume_l": ("예측에 사용", "주문 시점에 확정되는 상품 총부피"),
    "primary_category": ("예측에 사용", "주문 시점에 확정되는 대표 상품 카테고리"),
    "purchase_weekday": ("예측에 사용", "주문 시점 타임스탬프에서 파생된 요일"),
    "purchase_hour": ("예측에 사용", "주문 시점 타임스탬프에서 파생된 시각"),
}


def main() -> None:
    cols = pd.read_csv(RAW_PATH, nrows=0).columns.tolist()

    missing_in_map = [c for c in cols if c not in ROLE_MAP]
    extra_in_map = [c for c in ROLE_MAP if c not in cols]
    if missing_in_map or extra_in_map:
        raise ValueError(
            f"열 목록 불일치 - 파일에만 있음: {missing_in_map}, 매핑에만 있음: {extra_in_map}"
        )

    table = pd.DataFrame(
        [{"열이름": c, "구분": ROLE_MAP[c][0], "사유": ROLE_MAP[c][1]} for c in cols]
    )
    order = ["식별자", "목표 변수", "예측에 사용", "예측에서 제외"]
    table["구분"] = pd.Categorical(table["구분"], categories=order, ordered=True)
    table = table.sort_values("구분").reset_index(drop=True)

    print("=" * 78)
    print("변수 역할 분류표 (목표 변수: delivery_days)")
    print("=" * 78)
    print(table.to_string(index=False))

    n_features = int((table["구분"] == "예측에 사용").sum())
    print(f"\n예측에 사용할 변수 수: {n_features}개")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "03_variable_roles.csv"
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"저장: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
