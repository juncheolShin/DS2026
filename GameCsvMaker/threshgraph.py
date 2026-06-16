import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 파일 설정
# =========================

input_csv_file = "game_attribute_scores_with_overall_ratio.csv"

output_dir = "threshold_visualizations"
summary_output_file = "threshold_group_comparison_summary.csv"

os.makedirs(output_dir, exist_ok=True)

# =========================
# 한글 폰트 설정
# =========================

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 컬럼 설정
# =========================

Y_COL = "overall_positive_review_ratio"

TAGS = [
    "최적화#프레임",
    "최적화#조작감",
    "시스템#버그",
    "콘텐츠#볼륨",
    "콘텐츠#몰입도",
    "시스템#밸런스",
    "시스템#독창성",
    "시스템#자유도",
    "UX#그래픽",
    "UX#캐릭터디자인",
    "UX#사운드",
    "스토리#내러티브",
    "운영#핵/치트",
    "운영#업데이트",
    "콘텐츠#난이도",
    "시스템#진입장벽",
    "콘텐츠#피로도",
]

# =========================
# threshold 조건
# =========================

MIN_TOTAL_GAMES = 20

# 임계치 기준 양쪽 집단 최소 게임 수
MIN_SAFE_GAMES = 7
MIN_RISK_GAMES = 7

# 전체 대비 최소 비율
MIN_SAFE_RATIO = 0.20
MIN_RISK_RATIO = 0.20

# 이 정도 이상 차이나야 의미 있어 보인다고 볼 기준
# 0.10 = 전체 긍정 리뷰 비율 10%p 차이
MIN_MEAN_GAP = 0.10

# =========================
# 유틸 함수
# =========================

def safe_file_name(name):
    return (
        name.replace("#", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace("*", "_")
            .replace("?", "_")
            .replace('"', "_")
            .replace("<", "_")
            .replace(">", "_")
            .replace("|", "_")
    )


def required_count(total_count, min_count, min_ratio):
    return max(min_count, math.ceil(total_count * min_ratio))


def cohen_d(a, b):
    """
    두 집단 평균 차이의 표준화 크기.
    a: 양호 집단
    b: 위험 집단
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    n1 = len(a)
    n2 = len(b)

    if n1 < 2 or n2 < 2:
        return np.nan

    var1 = np.var(a, ddof=1)
    var2 = np.var(b, ddof=1)

    pooled = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

    if pooled <= 0:
        return np.nan

    return (np.mean(a) - np.mean(b)) / np.sqrt(pooled)


def get_dynamic_reversed_xlim(x):
    """
    x축:
        왼쪽 = 긍정
        오른쪽 = 부정
    """
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)

    padding = (x_max - x_min) * 0.1

    if padding == 0 or pd.isna(padding):
        padding = 0.1

    left_limit = min(x_max + padding, 1.05)
    right_limit = max(x_min - padding, -1.05)

    return left_limit, right_limit


def find_best_threshold(x, y):
    """
    x = 속성 점수
        1 = 긍정
        0 = 중립
       -1 = 부정

    threshold t 기준:
        위험 집단 = x <= t
        양호 집단 = x > t

    목표:
        양호 집단 평균 평점 - 위험 집단 평균 평점이 가장 큰 t를 찾음.
    """

    total_count = len(x)

    required_safe = required_count(
        total_count,
        MIN_SAFE_GAMES,
        MIN_SAFE_RATIO
    )

    required_risk = required_count(
        total_count,
        MIN_RISK_GAMES,
        MIN_RISK_RATIO
    )

    unique_x = np.sort(np.unique(x))

    if len(unique_x) < 3:
        return None, pd.DataFrame()

    # 실제 점수 사이의 중간값을 threshold 후보로 사용
    candidate_thresholds = (unique_x[:-1] + unique_x[1:]) / 2

    rows = []

    for t in candidate_thresholds:
        risk_mask = x <= t
        safe_mask = x > t

        risk_y = y[risk_mask]
        safe_y = y[safe_mask]

        risk_n = len(risk_y)
        safe_n = len(safe_y)

        if risk_n < required_risk:
            continue

        if safe_n < required_safe:
            continue

        risk_mean = np.mean(risk_y)
        safe_mean = np.mean(safe_y)

        risk_median = np.median(risk_y)
        safe_median = np.median(safe_y)

        mean_gap = safe_mean - risk_mean
        median_gap = safe_median - risk_median

        d = cohen_d(safe_y, risk_y)

        rows.append({
            "threshold": t,
            "safe_n": safe_n,
            "risk_n": risk_n,
            "safe_mean": safe_mean,
            "risk_mean": risk_mean,
            "mean_gap": mean_gap,
            "safe_median": safe_median,
            "risk_median": risk_median,
            "median_gap": median_gap,
            "cohen_d": d,
        })

    scan_df = pd.DataFrame(rows)

    if scan_df.empty:
        return None, scan_df

    # 평균 차이가 가장 큰 threshold 선택
    best_idx = scan_df["mean_gap"].idxmax()
    best = scan_df.loc[best_idx].to_dict()

    return best, scan_df


# =========================
# CSV 읽기
# =========================

df = pd.read_csv(input_csv_file, encoding="utf-8-sig")

if "game_name" not in df.columns:
    raise ValueError("game_name 컬럼이 없습니다.")

if Y_COL not in df.columns:
    raise ValueError(f"{Y_COL} 컬럼이 없습니다.")

for tag in TAGS:
    if tag not in df.columns:
        raise ValueError(f"{tag} 컬럼이 없습니다.")

df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce")

for tag in TAGS:
    df[tag] = pd.to_numeric(df[tag], errors="coerce")

# 혹시 0~100이면 0~1로 변환
if df[Y_COL].max() > 1:
    print(f"{Y_COL} 값이 0~100 범위로 보입니다. 0~1로 변환합니다.")
    df[Y_COL] = df[Y_COL] / 100


# =========================
# 속성별 threshold 분석 및 시각화
# =========================

summary_rows = []
valid_tags = []

for tag in TAGS:
    plot_df = df[["game_name", tag, Y_COL]].dropna()

    if len(plot_df) < MIN_TOTAL_GAMES:
        print(f"[스킵] {tag}: 유효 게임 수 부족 ({len(plot_df)}개)")
        continue

    x = plot_df[tag].to_numpy(dtype=float)
    y = plot_df[Y_COL].to_numpy(dtype=float)

    best, scan_df = find_best_threshold(x, y)

    if best is None:
        print(f"[스킵] {tag}: 조건을 만족하는 threshold 없음")
        continue

    threshold = best["threshold"]

    risk_mask = x <= threshold
    safe_mask = x > threshold

    risk_y = y[risk_mask]
    safe_y = y[safe_mask]

    passes_gap = best["mean_gap"] >= MIN_MEAN_GAP

    interpretation = (
        "통과: 위험 집단 평점 하락 뚜렷"
        if passes_gap
        else "주의: 평균 차이 약함"
    )

    summary_rows.append({
        "attribute": tag,
        "valid_game_count": len(plot_df),
        "threshold_attribute_score": round(threshold, 4),

        "safe_count": int(best["safe_n"]),
        "risk_count": int(best["risk_n"]),

        "safe_mean_overall": round(best["safe_mean"], 4),
        "risk_mean_overall": round(best["risk_mean"], 4),
        "mean_gap": round(best["mean_gap"], 4),

        "safe_median_overall": round(best["safe_median"], 4),
        "risk_median_overall": round(best["risk_median"], 4),
        "median_gap": round(best["median_gap"], 4),

        "cohen_d": round(best["cohen_d"], 4) if not pd.isna(best["cohen_d"]) else np.nan,

        "passes_gap": passes_gap,
        "interpretation": interpretation,
    })

    valid_tags.append(tag)

    # =========================
    # 1. 산점도 + threshold 선 + 집단 평균선
    # =========================

    x_left, x_right = get_dynamic_reversed_xlim(x)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        x[safe_mask],
        y[safe_mask],
        alpha=0.7,
        label=f"양호 집단 n={len(safe_y)}"
    )

    plt.scatter(
        x[risk_mask],
        y[risk_mask],
        alpha=0.7,
        label=f"위험 집단 n={len(risk_y)}"
    )

    plt.axvline(
        threshold,
        linestyle="--",
        label=f"threshold = {threshold:.3f}"
    )

    plt.axhline(
        np.mean(safe_y),
        linestyle=":",
        label=f"양호 평균 = {np.mean(safe_y):.3f}"
    )

    plt.axhline(
        np.mean(risk_y),
        linestyle="-.",
        label=f"위험 평균 = {np.mean(risk_y):.3f}"
    )

    plt.title(f"{tag} threshold 기준 집단 분리")
    plt.xlabel(f"{tag} 감정 점수 (왼쪽 = 긍정, 오른쪽 = 부정)")
    plt.ylabel("전체 긍정 리뷰 비율")

    plt.xlim(x_left, x_right)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()

    info_text = (
        f"threshold: {threshold:.3f}\n"
        f"양호 평균: {np.mean(safe_y):.3f}\n"
        f"위험 평균: {np.mean(risk_y):.3f}\n"
        f"평균 차이: {best['mean_gap']:.3f}\n"
        f"{interpretation}"
    )

    plt.text(
        x_left,
        1.0,
        info_text,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    scatter_path = os.path.join(
        output_dir,
        f"{safe_file_name(tag)}_threshold_scatter.png"
    )

    plt.tight_layout()
    plt.savefig(scatter_path, dpi=300)
    plt.close()

    # =========================
    # 2. threshold 후보별 평균 차이 그래프
    # =========================

    plt.figure(figsize=(8, 5))

    plt.plot(
        scan_df["threshold"],
        scan_df["mean_gap"],
        marker="o"
    )

    plt.axvline(
        threshold,
        linestyle="--",
        label=f"best threshold = {threshold:.3f}"
    )

    plt.axhline(
        MIN_MEAN_GAP,
        linestyle=":",
        label=f"기준 평균 차이 = {MIN_MEAN_GAP:.2f}"
    )

    plt.title(f"{tag} threshold별 평균 평점 차이")
    plt.xlabel(f"{tag} threshold 점수")
    plt.ylabel("양호 집단 평균 - 위험 집단 평균")

    # threshold 축도 왼쪽 긍정, 오른쪽 부정
    tx_left, tx_right = get_dynamic_reversed_xlim(scan_df["threshold"].to_numpy())
    plt.xlim(tx_left, tx_right)

    plt.grid(True, alpha=0.3)
    plt.legend()

    scan_path = os.path.join(
        output_dir,
        f"{safe_file_name(tag)}_threshold_scan.png"
    )

    plt.tight_layout()
    plt.savefig(scan_path, dpi=300)
    plt.close()

    # =========================
    # 3. 박스플롯
    # =========================

    plt.figure(figsize=(7, 5))

    plt.boxplot(
        [safe_y, risk_y],
        labels=[
            f"양호 집단\nx > {threshold:.3f}\nn={len(safe_y)}",
            f"위험 집단\nx <= {threshold:.3f}\nn={len(risk_y)}"
        ],
        showmeans=True
    )

    plt.title(f"{tag} threshold 기준 전체 평점 분포")
    plt.ylabel("전체 긍정 리뷰 비율")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", alpha=0.3)

    box_path = os.path.join(
        output_dir,
        f"{safe_file_name(tag)}_threshold_boxplot.png"
    )

    plt.tight_layout()
    plt.savefig(box_path, dpi=300)
    plt.close()

    print(
        f"[완료] {tag} | "
        f"threshold={threshold:.4f}, "
        f"gap={best['mean_gap']:.4f}, "
        f"safe_n={int(best['safe_n'])}, "
        f"risk_n={int(best['risk_n'])}, "
        f"{interpretation}"
    )


# =========================
# 요약 CSV 저장
# =========================

summary_df = pd.DataFrame(summary_rows)

if not summary_df.empty:
    summary_df = summary_df.sort_values(
        ["passes_gap", "mean_gap", "cohen_d"],
        ascending=[False, False, False],
        na_position="last"
    )

summary_df.to_csv(summary_output_file, index=False, encoding="utf-8-sig")

print(f"[저장 완료] {summary_output_file}")
print("완료!")