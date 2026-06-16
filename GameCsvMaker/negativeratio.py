import pandas as pd

attribute_file = "game_attribute_negative_ratios.csv"
summary_file = "game_review_summary.csv"
output_file = "game_attribute_negative_ratios_with_overall_ratio.csv"

attribute_df = pd.read_csv(attribute_file, encoding="utf-8-sig")
summary_df = pd.read_csv(summary_file, encoding="utf-8-sig")

attribute_df["game_name"] = attribute_df["game_name"].astype(str).str.strip()
summary_df["game_name"] = summary_df["game_name"].astype(str).str.strip()

summary_df = summary_df.drop_duplicates(subset=["game_name"], keep="first")

summary_cols = [
    "game_name",
    "overall_positive_review_ratio"
]

merged_df = pd.merge(
    attribute_df,
    summary_df[summary_cols],
    on="game_name",
    how="left"
)

# overall_positive_review_ratio를 game_name 바로 뒤로 이동
cols = list(merged_df.columns)
cols.remove("overall_positive_review_ratio")
cols.insert(1, "overall_positive_review_ratio")
merged_df = merged_df[cols]

merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"저장 완료: {output_file}")