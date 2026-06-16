import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 파일 설정
# =========================

input_csv_file = "game_attribute_scores_with_overall_ratio.csv"

output_dir = "threshold_piecewise_visualizations"
summary_output_file = "threshold_piecewise_summary.csv"
combined_output_file = "all_threshold_piecewise_scatter.png"

os.makedirs(output_dir, exist_ok=True)

# =========================
# 한글 폰트 설정
# Windows 기준
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

# threshold 양쪽 최소 게임 수
MIN_SAFE_GAMES = 7
MIN_RISK_GAMES = 7

# threshold 양쪽 최소 비율
MIN_SAFE_RATIO = 0.20
MIN_RISK_RATIO = 0.20

# 평균 차이가 이 값 이상이면 의미 있는 분리로 표시
# 0.10 = 전체 긍정 리뷰 비율 10%p 차이
MIN_MEAN_GAP = 0.10

# 구간별 회귀선을 그리기 위한 최소 조건
MIN_REGRESSION_GAMES = 3
MIN_UNIQUE_X_FOR_REGRESSION = 2

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


def r2_score(y, y_pred):
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rss = np.sum((y - y_pred) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)

    if tss == 0:
        return np.nan

    return 1 - rss / tss


def fit_simple_regression(x, y):
    """
    단순 선형회귀:
        y = intercept + slope * x

    x값 종류가 너무 적거나 데이터가 부족하면 None 반환.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < MIN_REGRESSION_GAMES:
        return None

    if len(np.unique(x)) < MIN_UNIQUE_X_FOR_REGRESSION:
        return None

    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    r2 = r2_score(y, y_pred)

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
    }


def regression_line_points(x, model):
    """
    회귀선 표시용 x/y 좌표 생성.
    """
    if model is None:
        return None, None

    x_min = np.min(x)
    x_max = np.max(x)

    if x_min == x_max:
        return None, None

    x_grid = np.linspace(x_min, x_max, 100)
    y_grid = model["slope"] * x_grid + model["intercept"]

    return x_grid, y_grid


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


def fmt(value, digits=3):
    if value is None:
        return "계산 불가"
    if pd.isna(value):
        return "계산 불가"
    return f"{value:.{digits}f}"


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
# 속성별 threshold + 구간별 회귀 분석
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

    best, scan_df = find_best_threshold(x, y)

    if best is None:
        print(f"[스킵] {tag}: 조건을 만족하는 threshold 없음")
        continue

    threshold = best["threshold"]

    risk_mask = x <= threshold
    safe_mask = x > threshold

    risk_x = x[risk_mask]
    risk_y = y[risk_mask]

    safe_x = x[safe_mask]
    safe_y = y[safe_mask]

    # threshold 기준 양쪽에 따로 선형회귀
    safe_model = fit_simple_regression(safe_x, safe_y)
    risk_model = fit_simple_regression(risk_x, risk_y)

    safe_slope = safe_model["slope"] if safe_model is not None else np.nan
    safe_r2 = safe_model["r2"] if safe_model is not None else np.nan

    risk_slope = risk_model["slope"] if risk_model is not None else np.nan
    risk_r2 = risk_model["r2"] if risk_model is not None else np.nan

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

        "safe_regression_slope": round(safe_slope, 4) if not pd.isna(safe_slope) else np.nan,
        "safe_regression_r2": round(safe_r2, 4) if not pd.isna(safe_r2) else np.nan,

        "risk_regression_slope": round(risk_slope, 4) if not pd.isna(risk_slope) else np.nan,
        "risk_regression_r2": round(risk_r2, 4) if not pd.isna(risk_r2) else np.nan,

        "passes_gap": passes_gap,
        "interpretation": interpretation,
    })

    valid_results[tag] = {
        "plot_df": plot_df,
        "x": x,
        "y": y,
        "threshold": threshold,
        "scan_df": scan_df,
        "safe_mask": safe_mask,
        "risk_mask": risk_mask,
        "safe_x": safe_x,
        "safe_y": safe_y,
        "risk_x": risk_x,
        "risk_y": risk_y,
        "safe_model": safe_model,
        "risk_model": risk_model,
        "best": best,
        "passes_gap": passes_gap,
        "interpretation": interpretation,
    }

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


# =========================
# 개별 시각화 저장
# =========================

for tag, result in valid_results.items():
    x = result["x"]
    y = result["y"]

    threshold = result["threshold"]

    safe_x = result["safe_x"]
    safe_y = result["safe_y"]
    risk_x = result["risk_x"]
    risk_y = result["risk_y"]

    safe_model = result["safe_model"]
    risk_model = result["risk_model"]

    best = result["best"]
    interpretation = result["interpretation"]

    x_left, x_right = get_dynamic_reversed_xlim(x)

    safe_line_x, safe_line_y = regression_line_points(safe_x, safe_model)
    risk_line_x, risk_line_y = regression_line_points(risk_x, risk_model)

    # =========================
    # 1. 산점도 + threshold + 구간별 회귀선
    # =========================

    plt.figure(figsize=(10, 6))

    plt.scatter(
        safe_x,
        safe_y,
        alpha=0.7,
        label=f"양호 집단 n={len(safe_y)}"
    )

    plt.scatter(
        risk_x,
        risk_y,
        alpha=0.7,
        label=f"위험 집단 n={len(risk_y)}"
    )

    if safe_line_x is not None:
        plt.plot(
            safe_line_x,
            safe_line_y,
            linestyle="-",
            label=f"양호 구간 회귀선, R²={fmt(safe_model['r2'])}"
        )

    if risk_line_x is not None:
        plt.plot(
            risk_line_x,
            risk_line_y,
            linestyle="-",
            label=f"위험 구간 회귀선, R²={fmt(risk_model['r2'])}"
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

    plt.title(f"{tag} threshold + 구간별 선형회귀")
    plt.xlabel(f"{tag} 감정 점수 (왼쪽 = 긍정, 오른쪽 = 부정)")
    plt.ylabel("전체 긍정 리뷰 비율")

    plt.xlim(x_left, x_right)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()

    safe_slope_text = fmt(safe_model["slope"]) if safe_model is not None else "계산 불가"
    risk_slope_text = fmt(risk_model["slope"]) if risk_model is not None else "계산 불가"

    info_text = (
        f"threshold: {threshold:.3f}\n"
        f"양호 평균: {np.mean(safe_y):.3f}\n"
        f"위험 평균: {np.mean(risk_y):.3f}\n"
        f"평균 차이: {best['mean_gap']:.3f}\n"
        f"양호 기울기: {safe_slope_text}\n"
        f"위험 기울기: {risk_slope_text}\n"
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

    scatter_path = os.path.join(
        output_dir,
        f"{safe_file_name(tag)}_threshold_piecewise_scatter.png"
    )

    plt.subplots_adjust(right=0.75)
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close()

    # =========================
    # 2. threshold 후보별 평균 차이 그래프
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
    # 3. 박스플롯
    # =========================

    plt.figure(figsize=(7, 5))

    plt.boxplot(
        [safe_y, risk_y],
        tick_labels=[
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


# =========================
# 전체 산점도 + 구간별 회귀 한 장으로 저장
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

        safe_x = result["safe_x"]
        safe_y = result["safe_y"]
        risk_x = result["risk_x"]
        risk_y = result["risk_y"]

        safe_model = result["safe_model"]
        risk_model = result["risk_model"]

        best = result["best"]
        passes_gap = result["passes_gap"]

        x_left, x_right = get_dynamic_reversed_xlim(x)

        safe_line_x, safe_line_y = regression_line_points(safe_x, safe_model)
        risk_line_x, risk_line_y = regression_line_points(risk_x, risk_model)

        ax.scatter(safe_x, safe_y, alpha=0.7)
        ax.scatter(risk_x, risk_y, alpha=0.7)

        if safe_line_x is not None:
            ax.plot(safe_line_x, safe_line_y, linestyle="-")

        if risk_line_x is not None:
            ax.plot(risk_line_x, risk_line_y, linestyle="-")

        ax.axvline(threshold, linestyle="--")

        status = "PASS" if passes_gap else "WEAK"

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