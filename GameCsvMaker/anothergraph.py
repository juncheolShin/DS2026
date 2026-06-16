import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 파일 설정
# =========================

input_csv_file = "game_attribute_scores_with_overall_ratio.csv"

output_dir = "piecewise_regression_plots"
summary_output_file = "piecewise_threshold_summary.csv"
combined_output_file = "all_piecewise_regression_plots.png"

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
# 분석 조건
# =========================

# 해당 속성 점수가 있는 게임이 최소 몇 개 이상 있어야 분석할지
MIN_TOTAL_GAMES = 20

# 임계치 이전, 즉 덜 부정적인 구간의 최소 게임 수
MIN_PRE_GAMES = 7

# 임계치 이후, 즉 더 부정적인 구간의 최소 게임 수
MIN_POST_GAMES = 7

# 임계치 이전/이후가 전체의 최소 몇 % 이상이어야 하는지
MIN_PRE_RATIO = 0.20
MIN_POST_RATIO = 0.20

# 구간별 회귀가 단순 선형회귀보다 RSS를 최소 몇 % 줄여야 통과인지
MIN_IMPROVEMENT_RATIO = 0.08

# 구간별 회귀의 최소 R²
MIN_PIECEWISE_R2 = 0.10

# 임계치 이후 기울기가 이전보다 최소 이 정도는 더 하락해야 함
# badness 기준이므로, post_slope가 pre_slope보다 더 작아져야 함
MIN_SLOPE_DROP = 0.05

# True면 품질 기준을 통과한 속성만 그래프 저장
# False면 탈락한 속성도 그래프 저장하고, CSV에서 통과 여부 확인
DRAW_ONLY_PASSED = False


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


def r2_score(y, y_pred):
    rss = np.sum((y - y_pred) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)

    if tss == 0:
        return np.nan

    return 1 - rss / tss


def get_dynamic_reversed_xlim(x_score):
    """
    그래프 표시용 x축 범위.

    속성 점수:
        1  = 긍정
        0  = 중립
       -1  = 부정

    그래프는 왼쪽이 긍정, 오른쪽이 부정이 되도록 x축을 역순으로 둔다.
    """

    x_min = np.nanmin(x_score)
    x_max = np.nanmax(x_score)

    padding = (x_max - x_min) * 0.1

    if padding == 0 or pd.isna(padding):
        padding = 0.1

    left_limit = x_max + padding
    right_limit = x_min - padding

    left_limit = min(left_limit, 1.05)
    right_limit = max(right_limit, -1.05)

    return left_limit, right_limit


def fit_linear_model(z, y):
    """
    단순 선형회귀:
        y = b0 + b1 * z

    z = badness
        값이 클수록 속성 평가가 나쁨
    """

    X = np.column_stack([
        np.ones(len(z)),
        z
    ])

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_pred = X @ beta

    rss = np.sum((y - y_pred) ** 2)
    r2 = r2_score(y, y_pred)

    return {
        "beta": beta,
        "y_pred": y_pred,
        "rss": rss,
        "r2": r2,
        "slope": beta[1],
        "intercept": beta[0],
    }


def fit_piecewise_for_tau(z, y, tau):
    """
    연속 구간별 선형회귀:
        y = b0 + b1 * z + b2 * max(0, z - tau)

    z <= tau:
        y = b0 + b1 * z

    z > tau:
        y = b0 + b1 * z + b2 * (z - tau)
        기울기 = b1 + b2

    tau가 임계치다.
    """

    hinge = np.maximum(0, z - tau)

    X = np.column_stack([
        np.ones(len(z)),
        z,
        hinge
    ])

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_pred = X @ beta

    rss = np.sum((y - y_pred) ** 2)
    r2 = r2_score(y, y_pred)

    pre_slope = beta[1]
    post_slope = beta[1] + beta[2]

    return {
        "tau": tau,
        "beta": beta,
        "y_pred": y_pred,
        "rss": rss,
        "r2": r2,
        "pre_slope": pre_slope,
        "post_slope": post_slope,
        "slope_change": beta[2],
    }


def required_side_count(total_count, min_count, min_ratio):
    """
    최소 개수 조건과 최소 비율 조건을 동시에 적용한다.
    예:
        total_count = 40
        min_count = 7
        min_ratio = 0.2

        required = max(7, ceil(40 * 0.2)) = 8
    """
    return max(min_count, math.ceil(total_count * min_ratio))


def find_best_piecewise_breakpoint(
    z,
    y,
    min_pre_games=7,
    min_post_games=7,
    min_pre_ratio=0.20,
    min_post_ratio=0.20
):
    """
    가능한 breakpoint 후보를 전부 시험해서 RSS가 가장 낮은 tau를 선택한다.

    z = badness
        값이 클수록 속성 평가가 나쁨

    z <= tau:
        임계치 이전, 덜 부정적인 구간

    z > tau:
        임계치 이후, 더 부정적인 구간

    임계치 이후 구간에도 충분한 게임이 있어야 후보로 인정한다.
    """

    unique_z = np.sort(np.unique(z))
    total_count = len(z)

    if len(unique_z) < 3:
        return None

    candidate_taus = (unique_z[:-1] + unique_z[1:]) / 2

    required_pre = required_side_count(
        total_count,
        min_pre_games,
        min_pre_ratio
    )

    required_post = required_side_count(
        total_count,
        min_post_games,
        min_post_ratio
    )

    best_result = None

    for tau in candidate_taus:
        pre_count = np.sum(z <= tau)
        post_count = np.sum(z > tau)

        if pre_count < required_pre:
            continue

        if post_count < required_post:
            continue

        result = fit_piecewise_for_tau(z, y, tau)

        if best_result is None or result["rss"] < best_result["rss"]:
            best_result = result
            best_result["pre_count"] = int(pre_count)
            best_result["post_count"] = int(post_count)
            best_result["required_pre_count"] = int(required_pre)
            best_result["required_post_count"] = int(required_post)

    return best_result


def predict_piecewise(z_grid, beta, tau):
    hinge = np.maximum(0, z_grid - tau)

    X = np.column_stack([
        np.ones(len(z_grid)),
        z_grid,
        hinge
    ])

    return X @ beta


def format_float(value, digits=3):
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


# =========================
# 숫자 변환
# =========================

df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce")

for tag in TAGS:
    df[tag] = pd.to_numeric(df[tag], errors="coerce")


# =========================
# overall_positive_review_ratio 보정
# 0~100이면 0~1로 변환
# =========================

if df[Y_COL].max() > 1:
    print(f"{Y_COL} 값이 0~100 범위로 보입니다. 0~1 범위로 변환합니다.")
    df[Y_COL] = df[Y_COL] / 100


# =========================
# 속성별 구간별 회귀 수행
# =========================

summary_rows = []
valid_results = {}

for tag in TAGS:
    plot_df = df[["game_name", tag, Y_COL]].dropna()

    if len(plot_df) < MIN_TOTAL_GAMES:
        print(f"[스킵] {tag}: 유효 게임 수 부족 ({len(plot_df)}개)")
        continue

    # attribute_score:
    #   1 = 긍정, 0 = 중립, -1 = 부정
    x_score = plot_df[tag].to_numpy(dtype=float)

    # badness:
    #   값이 클수록 속성 평가가 나쁨
    z_badness = -x_score

    y = plot_df[Y_COL].to_numpy(dtype=float)

    if len(np.unique(z_badness)) < 3:
        print(f"[스킵] {tag}: 서로 다른 x값이 너무 적음")
        continue

    linear_result = fit_linear_model(z_badness, y)

    piecewise_result = find_best_piecewise_breakpoint(
        z_badness,
        y,
        min_pre_games=MIN_PRE_GAMES,
        min_post_games=MIN_POST_GAMES,
        min_pre_ratio=MIN_PRE_RATIO,
        min_post_ratio=MIN_POST_RATIO
    )

    if piecewise_result is None:
        print(f"[스킵] {tag}: 조건을 만족하는 안정적인 breakpoint 없음")
        continue

    linear_rss = linear_result["rss"]
    piecewise_rss = piecewise_result["rss"]

    if linear_rss == 0:
        improvement_ratio = np.nan
    else:
        improvement_ratio = (linear_rss - piecewise_rss) / linear_rss

    threshold_badness = piecewise_result["tau"]
    threshold_attribute_score = -threshold_badness

    pre_slope = piecewise_result["pre_slope"]
    post_slope = piecewise_result["post_slope"]
    piecewise_r2 = piecewise_result["r2"]

    # badness 기준:
    # badness 증가 = 속성 평가 악화
    # post_slope가 pre_slope보다 충분히 더 작아야
    # 임계치 이후 하락 강화라고 볼 수 있다.
    slope_drop = pre_slope - post_slope

    passes_improvement = (
        not pd.isna(improvement_ratio)
        and improvement_ratio >= MIN_IMPROVEMENT_RATIO
    )

    passes_r2 = (
        not pd.isna(piecewise_r2)
        and piecewise_r2 >= MIN_PIECEWISE_R2
    )

    passes_slope = slope_drop >= MIN_SLOPE_DROP

    passes_quality = (
        passes_improvement
        and passes_r2
        and passes_slope
    )

    if not passes_improvement:
        interpretation = "탈락: 단순 선형회귀 대비 개선 약함"
    elif not passes_r2:
        interpretation = "탈락: 구간별 회귀 설명력 낮음"
    elif not passes_slope:
        interpretation = "탈락: 임계치 이후 하락 강화 부족"
    else:
        interpretation = "통과: 임계치 이후 하락 강화"

    summary_rows.append({
        "attribute": tag,
        "valid_game_count": len(plot_df),

        "threshold_attribute_score": round(threshold_attribute_score, 4),
        "threshold_badness": round(threshold_badness, 4),

        "pre_threshold_count": piecewise_result["pre_count"],
        "post_threshold_count": piecewise_result["post_count"],
        "required_pre_count": piecewise_result["required_pre_count"],
        "required_post_count": piecewise_result["required_post_count"],

        "linear_r2": round(linear_result["r2"], 4) if not pd.isna(linear_result["r2"]) else np.nan,
        "piecewise_r2": round(piecewise_result["r2"], 4) if not pd.isna(piecewise_result["r2"]) else np.nan,
        "rss_improvement_ratio": round(improvement_ratio, 4) if not pd.isna(improvement_ratio) else np.nan,

        "pre_threshold_slope_badness": round(pre_slope, 4),
        "post_threshold_slope_badness": round(post_slope, 4),
        "slope_drop": round(slope_drop, 4),

        "passes_improvement": passes_improvement,
        "passes_r2": passes_r2,
        "passes_slope": passes_slope,
        "passes_quality": passes_quality,

        "interpretation": interpretation,
    })

    if DRAW_ONLY_PASSED and not passes_quality:
        print(f"[그래프 제외] {tag}: {interpretation}")
        continue

    valid_results[tag] = {
        "plot_df": plot_df,
        "x_score": x_score,
        "z_badness": z_badness,
        "y": y,
        "linear_result": linear_result,
        "piecewise_result": piecewise_result,
        "threshold_attribute_score": threshold_attribute_score,
        "threshold_badness": threshold_badness,
        "improvement_ratio": improvement_ratio,
        "slope_drop": slope_drop,
        "passes_quality": passes_quality,
        "interpretation": interpretation,
    }

    print(
        f"[완료] {tag} | "
        f"임계 속성점수={threshold_attribute_score:.4f}, "
        f"post_count={piecewise_result['post_count']}, "
        f"piecewise R²={piecewise_result['r2']:.4f}, "
        f"개선율={improvement_ratio:.4f}, "
        f"{interpretation}"
    )


# =========================
# 요약 CSV 저장
# =========================

summary_df = pd.DataFrame(summary_rows)

if not summary_df.empty:
    summary_df = summary_df.sort_values(
        ["passes_quality", "rss_improvement_ratio", "piecewise_r2"],
        ascending=[False, False, False],
        na_position="last"
    )

summary_df.to_csv(summary_output_file, index=False, encoding="utf-8-sig")
print(f"[저장 완료] {summary_output_file}")


# =========================
# 개별 그래프 저장
# =========================

for tag, result in valid_results.items():
    x_score = result["x_score"]
    z_badness = result["z_badness"]
    y = result["y"]

    piecewise_result = result["piecewise_result"]
    linear_result = result["linear_result"]

    tau = piecewise_result["tau"]
    beta_piecewise = piecewise_result["beta"]

    threshold_attribute_score = result["threshold_attribute_score"]
    improvement_ratio = result["improvement_ratio"]
    slope_drop = result["slope_drop"]
    interpretation = result["interpretation"]
    passes_quality = result["passes_quality"]

    x_left, x_right = get_dynamic_reversed_xlim(x_score)

    z_min = z_badness.min()
    z_max = z_badness.max()

    z_grid = np.linspace(z_min, z_max, 200)

    y_piecewise_grid = predict_piecewise(
        z_grid,
        beta_piecewise,
        tau
    )

    beta_linear = linear_result["beta"]
    y_linear_grid = beta_linear[0] + beta_linear[1] * z_grid

    # 그래프에서는 x축을 attribute_score로 보여준다.
    x_grid_score = -z_grid

    plt.figure(figsize=(8, 6))

    plt.scatter(x_score, y, alpha=0.7, label="게임")
    plt.plot(x_grid_score, y_linear_grid, linestyle=":", label="일반 선형회귀")
    plt.plot(x_grid_score, y_piecewise_grid, linestyle="-", label="구간별 선형회귀")

    plt.axvline(
        threshold_attribute_score,
        linestyle="--",
        label=f"임계치: {threshold_attribute_score:.3f}"
    )

    plt.title(f"{tag} 구간별 회귀 임계치 분석")
    plt.xlabel(f"{tag} 감정 점수 (왼쪽 = 긍정, 오른쪽 = 부정)")
    plt.ylabel("전체 긍정 리뷰 비율")

    plt.xlim(x_left, x_right)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()

    info_text = (
        f"게임 수: {len(x_score)}\n"
        f"임계 속성점수: {threshold_attribute_score:.3f}\n"
        f"임계 이후 게임 수: {piecewise_result['post_count']}\n"
        f"선형 R²: {format_float(linear_result['r2'])}\n"
        f"구간 R²: {format_float(piecewise_result['r2'])}\n"
        f"RSS 개선율: {format_float(improvement_ratio)}\n"
        f"slope_drop: {format_float(slope_drop)}\n"
        f"{interpretation}"
    )

    plt.text(
        x_left,
        1.0,
        info_text,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    quality_suffix = "PASS" if passes_quality else "FAIL"

    output_path = os.path.join(
        output_dir,
        f"{safe_file_name(tag)}_piecewise_{quality_suffix}.png"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[저장 완료] {output_path}")


# =========================
# 전체 그래프 한 장으로 저장
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

        x_score = result["x_score"]
        z_badness = result["z_badness"]
        y = result["y"]

        piecewise_result = result["piecewise_result"]
        linear_result = result["linear_result"]

        tau = piecewise_result["tau"]
        beta_piecewise = piecewise_result["beta"]

        threshold_attribute_score = result["threshold_attribute_score"]
        improvement_ratio = result["improvement_ratio"]
        passes_quality = result["passes_quality"]

        x_left, x_right = get_dynamic_reversed_xlim(x_score)

        z_min = z_badness.min()
        z_max = z_badness.max()

        z_grid = np.linspace(z_min, z_max, 200)

        y_piecewise_grid = predict_piecewise(
            z_grid,
            beta_piecewise,
            tau
        )

        beta_linear = linear_result["beta"]
        y_linear_grid = beta_linear[0] + beta_linear[1] * z_grid

        x_grid_score = -z_grid

        ax.scatter(x_score, y, alpha=0.7)
        ax.plot(x_grid_score, y_linear_grid, linestyle=":")
        ax.plot(x_grid_score, y_piecewise_grid, linestyle="-")

        ax.axvline(
            threshold_attribute_score,
            linestyle="--"
        )

        status = "PASS" if passes_quality else "FAIL"

        ax.set_title(
            f"{tag} [{status}]\n"
            f"임계치={threshold_attribute_score:.3f}, "
            f"뒤쪽 n={piecewise_result['post_count']}, "
            f"개선율={format_float(improvement_ratio)}"
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