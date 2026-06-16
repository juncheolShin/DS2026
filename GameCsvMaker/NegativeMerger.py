import pandas as pd
import numpy as np

input_file = "game_names_with_scores.csv"
output_file = "game_attribute_negative_ratios.csv"
detail_output_file = "game_attribute_negative_ratio_details.csv"

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

MIN_TOTAL_REVIEWS = 30

df = pd.read_csv(input_file, encoding="utf-8-sig")

for tag in TAGS:
    df[tag] = pd.to_numeric(df[tag], errors="coerce")

result_rows = []
detail_rows = []

for game_name, group in df.groupby("game_name"):
    total_review_count = len(group)

    result_row = {
        "game_name": game_name
    }

    detail_row = {
        "game_name": game_name,
        "total_review_count": total_review_count
    }

    for tag in TAGS:
        values = group[tag]

        negative_count = (values == -1).sum()
        mention_count = values.notna().sum()

        if total_review_count >= MIN_TOTAL_REVIEWS:
            negative_ratio = negative_count / total_review_count
        else:
            negative_ratio = np.nan

        result_row[tag] = negative_ratio

        detail_row[f"{tag}_negative_count"] = negative_count
        detail_row[f"{tag}_mention_count"] = mention_count
        detail_row[f"{tag}_negative_ratio_all_reviews"] = negative_ratio

    result_rows.append(result_row)
    detail_rows.append(detail_row)

result_df = pd.DataFrame(result_rows)
detail_df = pd.DataFrame(detail_rows)

result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
detail_df.to_csv(detail_output_file, index=False, encoding="utf-8-sig")

print(f"저장 완료: {output_file}")
print(f"저장 완료: {detail_output_file}")