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

# 이 속성 점수를 가진 게임이 최소 몇 개 이상 있어야 구간별 회귀를 할지
MIN_TOTAL_GAMES = 12

# 임계치 좌우에 최소 몇 개의 게임이 있어야 하는지
MIN_SIDE_GAMES = 4

# 개선율이 이 값보다 작으면 "임계치 효과 약함"으로 해석
MIN_IMPROVEMENT_RATIO = 0.05


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
    그래프 표시용.
    x축은 속성 점수 그대로 사용한다.

    속성 점수:
      1  = 긍정
      0  = 중립
     -1  = 부정

    그래프에서는 왼쪽이 긍정, 오른쪽이 부정이 되도록 x축을 뒤집는다.
    """

    x_min = x_score.min()
    x_max = x_score.max()

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
    일반 선형 회귀:
    y = b0 + b1 * z

    z = badness
    badness가 커질수록 속성 평가가 나빠진다.
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
    연속 구간별 선형 회귀:
    y = b0 + b1 * z + b2 * max(0, z - tau)

    z <= tau:
        y = b0 + b1 * z

    z > tau:
        y = b0 + b1 * z + b2 * (z - tau)
        기울기 = b1 + b2

    여기서 tau가 임계치다.
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


def find_best_piecewise_breakpoint(z, y, min_side_games=4):
    """
    가능한 breakpoint 후보를 전부 시험해서 RSS가 가장 낮은 tau를 선택한다.

    tau는 실제 데이터 지점 사이의 중간값으로 잡는다.
    그래야 한 점에 임계치가 딱 걸리는 애매함이 줄어든다.
    """

    unique_z = np.sort(np.unique(z))

    if len(unique_z) < 3:
        return None

    candidate_taus = (unique_z[:-1] + unique_z[1:]) / 2

    best_result = None

    for tau in candidate_taus:
        left_count = np.sum(z <= tau)
        right_count = np.sum(z > tau)

        if left_count < min_side_games:
            continue

        if right_count < min_side_games:
            continue

        result = fit_piecewise_for_tau(z, y, tau)

        if best_result is None or result["rss"] < best_result["rss"]:
            best_result = result
            best_result["left_count"] = int(left_count)
            best_result["right_count"] = int(right_count)

    return best_result


def predict_piecewise(z_grid, beta, tau):
    hinge = np.maximum(0, z_grid - tau)

    X = np.column_stack([
        np.ones(len(z_grid)),
        z_grid,
        hinge
    ])

    return X @ beta


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
        min_side_games=MIN_SIDE_GAMES
    )

    if piecewise_result is None:
        print(f"[스킵] {tag}: 조건을 만족하는 breakpoint 없음")
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

    # badness 기준 해석:
    # badness가 커질수록 속성 평가가 나빠짐.
    # post_slope가 더 음수면 임계치 이후 평점 하락이 더 강해진 것.
    if pd.isna(improvement_ratio):
        interpretation = "판단 어려움"
    elif improvement_ratio < MIN_IMPROVEMENT_RATIO:
        interpretation = "임계치 효과 약함"
    elif post_slope < pre_slope:
        interpretation = "임계치 이후 하락 강화"
    else:
        interpretation = "임계치 이후 하락 강화 아님"

    summary_rows.append({
        "attribute": tag,
        "valid_game_count": len(plot_df),

        "threshold_attribute_score": round(threshold_attribute_score, 4),
        "threshold_badness": round(threshold_badness, 4),

        "left_positive_side_count": piecewise_result["left_count"],
        "right_negative_side_count": piecewise_result["right_count"],

        "linear_r2": round(linear_result["r2"], 4) if not pd.isna(linear_result["r2"]) else "",
        "piecewise_r2": round(piecewise_result["r2"], 4) if not pd.isna(piecewise_result["r2"]) else "",
        "rss_improvement_ratio": round(improvement_ratio, 4) if not pd.isna(improvement_ratio) else "",

        "pre_threshold_slope_badness": round(pre_slope, 4),
        "post_threshold_slope_badness": round(post_slope, 4),
        "slope_change": round(piecewise_result["slope_change"], 4),

        "interpretation": interpretation,
    })

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
        "interpretation": interpretation,
    }

    print(
        f"[완료] {tag} | "
        f"임계 속성점수={threshold_attribute_score:.4f}, "
        f"piecewise R²={piecewise_result['r2']:.4f}, "
        f"개선율={improvement_ratio:.4f}"
    )


# =========================
# 요약 CSV 저장
# =========================

summary_df = pd.DataFrame(summary_rows)

if not summary_df.empty:
    summary_df = summary_df.sort_values(
        ["rss_improvement_ratio", "piecewise_r2"],
        ascending=False
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
    interpretation = result["interpretation"]

    x_left, x_right = get_dynamic_reversed_xlim(x_score)

    # 예측선 만들기
    z_min = z_badness.min()
    z_max = z_badness.max()

    z_grid = np.linspace(z_min, z_max, 200)

    y_piecewise_grid = predict_piecewise(
        z_grid,
        beta_piecewise,
        tau
    )

    # 일반 선형 회귀선도 같이 표시
    beta_linear = linear_result["beta"]
    y_linear_grid = beta_linear[0] + beta_linear[1] * z_grid

    # 그래프에서는 x축을 attribute_score로 보여준다
    x_grid_score = -z_grid

    plt.figure(figsize=(8, 6))

    plt.scatter(x_score, y, alpha=0.7, label="게임")

    plt.plot(
        x_grid_score,
        y_linear_grid,
        linestyle=":",
        label="일반 선형회귀"
    )

    plt.plot(
        x_grid_score,
        y_piecewise_grid,
        linestyle="-",
        label="구간별 선형회귀"
    )

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
        f"선형 R²: {linear_result['r2']:.3f}\n"
        f"구간 R²: {piecewise_result['r2']:.3f}\n"
        f"RSS 개선율: {improvement_ratio:.3f}\n"
        f"{interpretation}"
    )

    plt.text(
        x_left,
        1.0,
        info_text,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    output_path = os.path.join(
        output_dir,
        f"{safe_file_name(tag)}_piecewise.png"
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

        ax.plot(
            x_grid_score,
            y_linear_grid,
            linestyle=":"
        )

        ax.plot(
            x_grid_score,
            y_piecewise_grid,
            linestyle="-"
        )

        ax.axvline(
            threshold_attribute_score,
            linestyle="--"
        )

        ax.set_title(
            f"{tag}\n"
            f"임계치={threshold_attribute_score:.3f}, "
            f"개선율={improvement_ratio:.3f}"
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