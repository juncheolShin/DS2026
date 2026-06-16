import pandas as pd

# =========================
# 설정
# =========================

input_csv_file = "game_names_with_scores.csv"

output_score_file = "game_attribute_sentiment_scores.csv"
output_detail_file = "game_attribute_sentiment_details.csv"

MIN_COUNT = 7  # 속성 언급 리뷰가 7개 이상일 때만 계산

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
# CSV 읽기
# =========================

df = pd.read_csv(input_csv_file, encoding="utf-8-sig")

if "game_name" not in df.columns:
    raise ValueError("CSV에 game_name 컬럼이 없습니다.")

for tag in TAGS:
    if tag not in df.columns:
        raise ValueError(f"CSV에 '{tag}' 컬럼이 없습니다.")


# =========================
# 태그 컬럼 숫자 변환
# 빈칸은 NaN으로 유지
# =========================

for tag in TAGS:
    df[tag] = pd.to_numeric(df[tag], errors="coerce")


# =========================
# 게임별 속성 감정 점수 계산
# =========================

score_rows = []
detail_rows = []

for game_name, group in df.groupby("game_name", sort=False):
    score_row = {
        "game_name": game_name
    }

    detail_row = {
        "game_name": game_name
    }

    for tag in TAGS:
        # 해당 속성이 언급된 리뷰만 사용
        mentioned_reviews = group[tag].dropna()

        mention_count = len(mentioned_reviews)
        positive_count = (mentioned_reviews == 1).sum()
        neutral_count = (mentioned_reviews == 0).sum()
        negative_count = (mentioned_reviews == -1).sum()

        # 상세 정보 저장
        detail_row[f"{tag}_mention_count"] = mention_count
        detail_row[f"{tag}_positive_count"] = positive_count
        detail_row[f"{tag}_neutral_count"] = neutral_count
        detail_row[f"{tag}_negative_count"] = negative_count

        if mention_count >= MIN_COUNT:
            # 핵심 공식:
            # positive는 +1, neutral은 0, negative는 -1로 보고 평균
            sentiment_score = mentioned_reviews.mean()
            score_row[tag] = round(sentiment_score, 4)
        else:
            # 언급 수 부족하면 빈칸
            score_row[tag] = ""

    score_rows.append(score_row)
    detail_rows.append(detail_row)


# =========================
# 결과 저장
# =========================

score_df = pd.DataFrame(score_rows)
detail_df = pd.DataFrame(detail_rows)

score_df.to_csv(output_score_file, index=False, encoding="utf-8-sig")
detail_df.to_csv(output_detail_file, index=False, encoding="utf-8-sig")

print("완료!")
print(f"속성 감정 점수 파일: {output_score_file}")
print(f"상세 카운트 파일: {output_detail_file}")