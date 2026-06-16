import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 파일 설정
# =========================

input_csv_file = "game_attribute_scores_with_overall_ratio.csv"

output_dir = "attribute_scatter_plots"
combined_output_file = "all_attribute_scatter_plots.png"
summary_output_file = "attribute_correlation_summary.csv"

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
# 유틸 함수
# =========================

def get_dynamic_reversed_xlim(x):
    """
    x축을 실제 데이터 최소~최대 범위로 잡되,
    왼쪽이 긍정, 오른쪽이 부정이 되도록 역순으로 반환한다.

    원래 점수:
        1  = 긍정
        0  = 중립
        -1 = 부정

    따라서:
        왼쪽 = x_max
        오른쪽 = x_min
    """

    x_min = x.min()
    x_max = x.max()

    padding = (x_max - x_min) * 0.1

    # 모든 x 값이 같은 경우
    if padding == 0 or pd.isna(padding):
        padding = 0.1

    left_limit = x_max + padding
    right_limit = x_min - padding

    # 점수 범위가 원래 -1 ~ 1이므로 너무 튀지 않게 제한
    left_limit = min(left_limit, 1.05)
    right_limit = max(right_limit, -1.05)

    return left_limit, right_limit


def safe_file_name(name):
    """
    파일명에 쓰기 어려운 문자 정리
    """
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
# 혹시 0~100 퍼센트면 0~1 비율로 변환
# =========================

if df[Y_COL].max() > 1:
    print(f"{Y_COL} 값이 0~100 범위로 보입니다. 0~1 범위로 변환합니다.")
    df[Y_COL] = df[Y_COL] / 100


# =========================
# 개별 산점도 생성
# =========================

summary_rows = []
valid_tags = []

for tag in TAGS:
    plot_df = df[["game_name", tag, Y_COL]].dropna()

    if len(plot_df) < 3:
        print(f"[스킵] {tag}: 데이터가 너무 적음 ({len(plot_df)}개)")
        continue

    x = plot_df[tag]
    y = plot_df[Y_COL]

    x_min = x.min()
    x_max = x.max()

    x_left, x_right = get_dynamic_reversed_xlim(x)

    # 상관계수
    corr = x.corr(y)

    # 추세선은 x 값이 최소 2종류 이상일 때만 계산
    can_draw_trend = x.nunique() >= 2

    if can_draw_trend:
        slope, intercept = np.polyfit(x, y, 1)
        trend_x = np.linspace(x_min, x_max, 100)
        trend_y = slope * trend_x + intercept
    else:
        slope = np.nan
        intercept = np.nan
        trend_x = None
        trend_y = None

    summary_rows.append({
        "attribute": tag,
        "valid_game_count": len(plot_df),
        "x_min": round(x_min, 4),
        "x_max": round(x_max, 4),
        "correlation": round(corr, 4) if not pd.isna(corr) else "",
        "slope": round(slope, 4) if not pd.isna(slope) else "",
        "intercept": round(intercept, 4) if not pd.isna(intercept) else "",
    })

    valid_tags.append(tag)

    plt.figure(figsize=(8, 6))

    plt.scatter(x, y, alpha=0.7)

    if can_draw_trend:
        plt.plot(trend_x, trend_y, linestyle="--")

    plt.title(f"{tag} 점수와 전체 긍정 리뷰 비율")
    plt.xlabel(f"{tag} 감정 점수 (왼쪽 = 긍정, 오른쪽 = 부정)")
    plt.ylabel("전체 긍정 리뷰 비율")

    # 핵심: x축 가변 범위 + 역순
    plt.xlim(x_left, x_right)

    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)

    plt.text(
        x_left,
        1.0,
        f"게임 수: {len(plot_df)}\n상관계수 r = {corr:.3f}" if not pd.isna(corr) else f"게임 수: {len(plot_df)}\n상관계수 r = 계산 불가",
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    output_path = os.path.join(output_dir, f"{safe_file_name(tag)}_scatter.png")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[저장 완료] {output_path}")


# =========================
# 상관계수 요약 CSV 저장
# =========================

summary_df = pd.DataFrame(summary_rows)

if not summary_df.empty:
    summary_df = summary_df.sort_values("correlation", ascending=False, na_position="last")

summary_df.to_csv(summary_output_file, index=False, encoding="utf-8-sig")

print(f"[저장 완료] {summary_output_file}")


# =========================
# 전체 속성 그래프를 한 장에 모으기
# =========================

if len(valid_tags) == 0:
    print("전체 그래프를 만들 수 없습니다. 유효한 속성 데이터가 없습니다.")
else:
    cols = 3
    rows = math.ceil(len(valid_tags) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))

    # 그래프가 1개일 때도 처리
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, tag in enumerate(valid_tags):
        ax = axes[idx]

        plot_df = df[["game_name", tag, Y_COL]].dropna()

        x = plot_df[tag]
        y = plot_df[Y_COL]

        x_min = x.min()
        x_max = x.max()

        x_left, x_right = get_dynamic_reversed_xlim(x)

        corr = x.corr(y)

        can_draw_trend = x.nunique() >= 2

        if can_draw_trend:
            slope, intercept = np.polyfit(x, y, 1)
            trend_x = np.linspace(x_min, x_max, 100)
            trend_y = slope * trend_x + intercept

        ax.scatter(x, y, alpha=0.7)

        if can_draw_trend:
            ax.plot(trend_x, trend_y, linestyle="--")

        if not pd.isna(corr):
            title = f"{tag}\nr = {corr:.3f}, n = {len(plot_df)}"
        else:
            title = f"{tag}\nr = 계산 불가, n = {len(plot_df)}"

        ax.set_title(title)
        ax.set_xlabel("속성 감정 점수\n왼쪽 = 긍정, 오른쪽 = 부정")
        ax.set_ylabel("전체 긍정 리뷰 비율")

        # 핵심: x축 가변 범위 + 역순
        ax.set_xlim(x_left, x_right)

        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    # 남는 빈 그래프 칸 제거
    for idx in range(len(valid_tags), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(combined_output_file, dpi=300)
    plt.close()

    print(f"[저장 완료] {combined_output_file}")

print("완료!")