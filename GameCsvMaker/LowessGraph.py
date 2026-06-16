import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# =========================
# 파일 설정
# =========================

input_csv_file = "game_attribute_scores_with_overall_ratio.csv"

output_dir = "lowess_visualizations"
summary_output_file = "lowess_threshold_summary.csv"
combined_output_file = "all_lowess_scatter.png"

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
# 분석 조건
# =========================

MIN_TOTAL_GAMES = 20

# threshold 양쪽 최소 게임 수
MIN_SAFE_GAMES = 7
MIN_RISK_GAMES = 7

# threshold 양쪽 최소 비율
MIN_SAFE_RATIO = 0.20
MIN_RISK_RATIO = 0.20

# 평균 차이가 이 값 이상이면 의미 있는 분리로 표시
MIN_MEAN_GAP = 0.10

# LOWESS 부드러움 정도
# 작을수록 구불구불, 클수록 부드러움
# 데이터가 군집형이면 0.4~0.6 추천
LOWESS_FRAC = 0.45

# LOWESS 반복 횟수
# 이상치 영향 줄이려면 1~3
LOWESS_IT = 1

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
            "cohen_d": cohen_d(safe_y, risk_y),
        })

    scan_df = pd.DataFrame(rows)

    if scan_df.empty:
        return None, scan_df

    best_idx = scan_df["mean_gap"].idxmax()
    best = scan_df.loc[best_idx].to_dict()

    return best, scan_df


def make_lowess_curve(x, y):
    """
    LOWESS 추세선 생성.
    반환값:
        lowess_x, lowess_y
    """
    if len(x) < MIN_TOTAL_GAMES:
        return None, None

    if len(np.unique(x)) < 3:
        return None, None

    result = lowess(
        endog=y,
        exog=x,
        frac=LOWESS_FRAC,
        it=LOWESS_IT,
        return_sorted=True
    )

    lowess_x = result[:, 0]
    lowess_y = result[:, 1]

    return lowess_x, lowess_y


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

# 0~100이면 0~1로 변환
if df[Y_COL].max() > 1:
    print(f"{Y_COL} 값이 0~100 범위로 보입니다. 0~1로 변환합니다.")
    df[Y_COL] = df[Y_COL] / 100


# =========================
# 속성별 LOWESS 시각화
# =========================

summary_rows = []
valid_results = {}

for tag in TAGS:
    plot_df = df[["game_name", tag, Y_COL]].dropna()

    if len(plot_df) < MIN_TOTAL_GAMES:
        print(f"[스킵] {tag}: 유효 게임 수 부족 ({len(plot_df)}개)")
        continue

    x = plot_df[tag].to_numpy(dtype=float)
    y = plot_df[Y_COL].to_numpy(dtype=float)

    if len(np.unique(x)) < 3:
        print(f"[스킵] {tag}: 서로 다른 x값이 너무 적음")
        continue

    best, scan_df = find_best_threshold(x, y)

    if best is None:
        print(f"[스킵] {tag}: 조건을 만족하는 threshold 없음")
        continue

    threshold = best["threshold"]

    risk_mask = x <= threshold
    safe_mask = x > threshold

    risk_y = y[risk_mask]
    safe_y = y[safe_mask]

    lowess_x, lowess_y = make_lowess_curve(x, y)

    if lowess_x is None:
        print(f"[스킵] {tag}: LOWESS 생성 불가")
        continue

    passes_gap = best["mean_gap"] >= MIN_MEAN_GAP

    interpretation = (
        "강함: threshold 기준 집단 차이 뚜렷"
        if passes_gap
        else "약함: threshold 기준 집단 차이 작음"
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

        "lowess_frac": LOWESS_FRAC,
        "passes_gap": passes_gap,
        "interpretation": interpretation,
    })

    valid_results[tag] = {
        "x": x,
        "y": y,
        "threshold": threshold,
        "risk_mask": risk_mask,
        "safe_mask": safe_mask,
        "lowess_x": lowess_x,
        "lowess_y": lowess_y,
        "scan_df": scan_df,
        "best": best,
        "passes_gap": passes_gap,
        "interpretation": interpretation,
    }

    print(
        f"[완료] {tag} | "
        f"threshold={threshold:.4f}, "
        f"gap={best['mean_gap']:.4f}, "
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


# =========================
# 개별 그래프 저장
# =========================

for tag, result in valid_results.items():
    x = result["x"]
    y = result["y"]

    threshold = result["threshold"]
    risk_mask = result["risk_mask"]
    safe_mask = result["safe_mask"]

    lowess_x = result["lowess_x"]
    lowess_y = result["lowess_y"]

    best = result["best"]
    interpretation = result["interpretation"]

    safe_y = y[safe_mask]
    risk_y = y[risk_mask]

    x_left, x_right = get_dynamic_reversed_xlim(x)

    plt.figure(figsize=(10, 6))

    plt.scatter(
        x[safe_mask],
        y[safe_mask],
        alpha=0.65,
        label=f"양호 집단 n={len(safe_y)}"
    )

    plt.scatter(
        x[risk_mask],
        y[risk_mask],
        alpha=0.65,
        label=f"위험 집단 n={len(risk_y)}"
    )

    plt.plot(
        lowess_x,
        lowess_y,
        linewidth=2.5,
        label=f"LOWESS 추세선 frac={LOWESS_FRAC}"
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

    plt.title(f"{tag} threshold + LOWESS 추세선")
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
        f"LOWESS frac: {LOWESS_FRAC}\n"
        f"{interpretation}"
    )

    ax = plt.gca()
    ax.text(
        1.02, 0.98,
        info_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    output_path = os.path.join(
        output_dir,
        f"{safe_file_name(tag)}_lowess_scatter.png"
    )

    plt.subplots_adjust(right=0.75)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # =========================
    # threshold scan 그래프도 저장
    # =========================

    scan_df = result["scan_df"]

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
# 전체 그래프 한 장 저장
# =========================

valid_tags = list(valid_results.keys())

if len(valid_tags) == 0:
    print("전체 그래프를 만들 수 없습니다. 유효한 속성 결과가 없습니다.")
else:
    cols = 3
    rows = math.ceil(len(valid_tags) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))

    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, tag in enumerate(valid_tags):
        ax = axes[idx]

        result = valid_results[tag]

        x = result["x"]
        y = result["y"]

        threshold = result["threshold"]
        risk_mask = result["risk_mask"]
        safe_mask = result["safe_mask"]

        lowess_x = result["lowess_x"]
        lowess_y = result["lowess_y"]

        best = result["best"]
        passes_gap = result["passes_gap"]

        x_left, x_right = get_dynamic_reversed_xlim(x)

        ax.scatter(x[safe_mask], y[safe_mask], alpha=0.65)
        ax.scatter(x[risk_mask], y[risk_mask], alpha=0.65)

        ax.plot(lowess_x, lowess_y, linewidth=2.2)

        ax.axvline(threshold, linestyle="--")

        status = "STRONG" if passes_gap else "WEAK"

        ax.set_title(
            f"{tag} [{status}]\n"
            f"t={threshold:.3f}, gap={best['mean_gap']:.3f}"
        )

        ax.set_xlabel("속성 감정 점수\n왼쪽 = 긍정, 오른쪽 = 부정")
        ax.set_ylabel("전체 긍정 리뷰 비율")

        ax.set_xlim(x_left, x_right)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    for idx in range(len(valid_tags), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(combined_output_file, dpi=300)
    plt.close()

    print(f"[저장 완료] {combined_output_file}")

print("완료!")