# -*- coding: utf-8 -*-
"""
data/raw/olist_delivery_orders_sample.csv 기반 탐색.

수행 내용
  1) delivery_days 평균/최소/최대 + 히스토그램
  2) distance_km vs total_weight_kg 산점도
  3) distance_km vs total_freight_brl 산점도
  4) 상관계수(Pearson) 계산 및 관계 해석용 수치 근거 출력

필요 라이브러리: pandas, matplotlib (표준 라이브러리: sys, pathlib)
원본 파일은 읽기만 하며 수정하지 않는다.
산점도의 결측 처리 기준과 제외 건수는 각 그림 제목/출력에 명시한다.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

RAW_PATH = Path("data/raw/olist_delivery_orders_sample.csv")
FIG_DIR = Path("outputs/figures")
TAB_DIR = Path("outputs/tables")


def main() -> None:
    df = pd.read_csv(RAW_PATH)
    n_rows = len(df)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1) delivery_days 기술통계 + 히스토그램 --------------------------
    dd = df["delivery_days"]
    dd_mean, dd_min, dd_max = dd.mean(), dd.min(), dd.max()
    print("=" * 78)
    print("[1] delivery_days 기술통계 (결측 0건, n=%d)" % dd.notna().sum())
    print(f"    평균 : {dd_mean:.3f} 일")
    print(f"    최소 : {dd_min:.3f} 일")
    print(f"    최대 : {dd_max:.3f} 일")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(dd, bins=50, color="#4C72B0", edgecolor="white")
    ax.axvline(dd_mean, color="#C44E52", linestyle="--", linewidth=1.5, label=f"평균 {dd_mean:.1f}일")
    ax.set_title(f"배송 소요일수(delivery_days) 분포 (n={n_rows:,})")
    ax.set_xlabel("delivery_days (일)")
    ax.set_ylabel("주문 건수")
    ax.legend()
    fig.tight_layout()
    hist_path = FIG_DIR / "02_delivery_days_hist.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"    저장: {hist_path.as_posix()}")

    # ---- 2) distance_km vs total_weight_kg 산점도 -------------------------
    cols_a = ["distance_km", "total_weight_kg"]
    sub_a = df[cols_a].dropna()
    n_drop_a = n_rows - len(sub_a)
    corr_a = sub_a["distance_km"].corr(sub_a["total_weight_kg"])
    print()
    print("[2] distance_km vs total_weight_kg")
    print(f"    결측 제외 건수: {n_drop_a}건 (distance_km 또는 total_weight_kg 결측) -> 사용 n={len(sub_a):,}")
    print(f"    Pearson 상관계수: {corr_a:.4f}")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(sub_a["distance_km"], sub_a["total_weight_kg"], s=8, alpha=0.25, color="#55A868")
    ax.set_title(f"거리 vs 총중량 산점도 (n={len(sub_a):,}, r={corr_a:.3f})")
    ax.set_xlabel("distance_km (km)")
    ax.set_ylabel("total_weight_kg (kg)")
    fig.tight_layout()
    sc_a_path = FIG_DIR / "02_scatter_distance_vs_weight.png"
    fig.savefig(sc_a_path, dpi=150)
    plt.close(fig)
    print(f"    저장: {sc_a_path.as_posix()}")

    # ---- 3) distance_km vs total_freight_brl 산점도 ------------------------
    cols_b = ["distance_km", "total_freight_brl"]
    sub_b = df[cols_b].dropna()
    n_drop_b = n_rows - len(sub_b)
    corr_b = sub_b["distance_km"].corr(sub_b["total_freight_brl"])
    print()
    print("[3] distance_km vs total_freight_brl")
    print(f"    결측 제외 건수: {n_drop_b}건 (distance_km 결측) -> 사용 n={len(sub_b):,}")
    print(f"    Pearson 상관계수: {corr_b:.4f}")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(sub_b["distance_km"], sub_b["total_freight_brl"], s=8, alpha=0.25, color="#C44E52")
    ax.set_title(f"거리 vs 배송비 산점도 (n={len(sub_b):,}, r={corr_b:.3f})")
    ax.set_xlabel("distance_km (km)")
    ax.set_ylabel("total_freight_brl (BRL)")
    fig.tight_layout()
    sc_b_path = FIG_DIR / "02_scatter_distance_vs_freight.png"
    fig.savefig(sc_b_path, dpi=150)
    plt.close(fig)
    print(f"    저장: {sc_b_path.as_posix()}")

    # ---- 요약 표 저장 -------------------------------------------------------
    summary = pd.DataFrame(
        [
            {"항목": "delivery_days 평균", "값": round(dd_mean, 3)},
            {"항목": "delivery_days 최소", "값": round(dd_min, 3)},
            {"항목": "delivery_days 최대", "값": round(dd_max, 3)},
            {"항목": "distance_km vs total_weight_kg 상관계수(n=%d)" % len(sub_a), "값": round(corr_a, 4)},
            {"항목": "distance_km vs total_freight_brl 상관계수(n=%d)" % len(sub_b), "값": round(corr_b, 4)},
        ]
    )
    out_path = TAB_DIR / "02_delivery_distance_summary.csv"
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    print()
    print(f"[4] 요약 저장: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
