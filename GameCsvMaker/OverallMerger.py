import pandas as pd

# =========================
# 파일 설정
# =========================

sentiment_score_file = "game_attribute_sentiment_scores.csv"
review_summary_file = "game_review_summary.csv"  # appid, game_name, overall_positive_review_ratio 있는 CSV
output_file = "game_attribute_scores_with_overall_ratio.csv"


# =========================
# CSV 읽기
# =========================

score_df = pd.read_csv(sentiment_score_file, encoding="utf-8-sig")
summary_df = pd.read_csv(review_summary_file, encoding="utf-8-sig")


# =========================
# 필수 컬럼 확인
# =========================

if "game_name" not in score_df.columns:
    raise ValueError("game_attribute_sentiment_scores.csv에 game_name 컬럼이 없습니다.")

if "game_name" not in summary_df.columns:
    raise ValueError("review summary CSV에 game_name 컬럼이 없습니다.")

if "overall_positive_review_ratio" not in summary_df.columns:
    raise ValueError("review summary CSV에 overall_positive_review_ratio 컬럼이 없습니다.")


# =========================
# game_name 정리
# 앞뒤 공백 때문에 매칭 안 되는 경우 방지
# =========================

score_df["game_name"] = score_df["game_name"].astype(str).str.strip()
summary_df["game_name"] = summary_df["game_name"].astype(str).str.strip()


# =========================
# 중복 game_name 검사
# =========================

score_duplicates = score_df[score_df["game_name"].duplicated(keep=False)]
summary_duplicates = summary_df[summary_df["game_name"].duplicated(keep=False)]

if not score_duplicates.empty:
    print("경고: 감정 점수 CSV에 중복 game_name이 있습니다.")
    print(score_duplicates["game_name"].value_counts().head(10))

if not summary_duplicates.empty:
    print("경고: 리뷰 요약 CSV에 중복 game_name이 있습니다.")
    print(summary_duplicates["game_name"].value_counts().head(10))
    print("일단 첫 번째 행만 사용합니다.")

    summary_df = summary_df.drop_duplicates(subset=["game_name"], keep="first")


# =========================
# 필요한 컬럼만 가져오기
# =========================

summary_ratio_df = summary_df[
    [
        "game_name",
        "overall_positive_review_ratio"
    ]
]


# =========================
# game_name 기준으로 병합
# =========================
# how="left":
# 감정 점수 CSV의 게임 목록을 기준으로 유지
# 순서도 score_df 기준으로 유지됨
# =========================

merged_df = score_df.merge(
    summary_ratio_df,
    on="game_name",
    how="left"
)


# =========================
# 매칭 안 된 게임 확인
# =========================

unmatched = merged_df[merged_df["overall_positive_review_ratio"].isna()]

if not unmatched.empty:
    print(f"매칭 안 된 게임 수: {len(unmatched)}")
    print("매칭 안 된 게임 예시:")
    print(unmatched["game_name"].head(20).to_string(index=False))
else:
    print("모든 게임이 정상 매칭되었습니다.")


# =========================
# 저장
# =========================

merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"완료! 생성된 파일: {output_file}")