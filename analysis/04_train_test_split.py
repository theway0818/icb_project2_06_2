# -*- coding: utf-8 -*-
"""
data/raw/olist_delivery_orders_sample.csv를 학습 80% / 테스트 20%로
무작위 분할한다.

- 분할 기준: sklearn train_test_split, test_size=0.2, random_state=42
- 행 제외/값 대체 없음 (전체 15,000행 그대로 분할)
- 결과는 data/processed/에 저장 (원본 data/raw/는 수정하지 않음)

필요 라이브러리: pandas, scikit-learn (표준 라이브러리: sys, pathlib)
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8")

RAW_PATH = Path("data/raw/olist_delivery_orders_sample.csv")
OUT_DIR = Path("data/processed")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main() -> None:
    df = pd.read_csv(RAW_PATH)
    n_total = len(df)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train.csv"
    test_path = OUT_DIR / "test.csv"
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print("=" * 78)
    print(f"전체 행 수: {n_total:,}")
    print(f"분할 기준 : test_size={TEST_SIZE}, random_state={RANDOM_STATE}")
    print("=" * 78)
    print(f"{'구분':<8}{'행 수':>10}{'비율':>10}{'delivery_days 평균':>22}")
    print(
        f"{'학습(train)':<8}{len(train_df):>10,}{len(train_df)/n_total*100:>9.2f}%"
        f"{train_df['delivery_days'].mean():>22.3f}"
    )
    print(
        f"{'테스트(test)':<8}{len(test_df):>10,}{len(test_df)/n_total*100:>9.2f}%"
        f"{test_df['delivery_days'].mean():>22.3f}"
    )
    print()
    print(f"저장: {train_path.as_posix()} ({len(train_df):,}행)")
    print(f"저장: {test_path.as_posix()} ({len(test_df):,}행)")


if __name__ == "__main__":
    main()
