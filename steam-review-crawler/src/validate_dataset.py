from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _format_timestamp(value: object) -> str:
    if pd.isna(value):
        return "NA"
    try:
        return str(pd.to_datetime(int(value), unit="s", utc=True))
    except Exception:
        return str(value)


def validate_dataset(
    reviews_path: str | Path,
    summary_path: str | Path | None = None,
    failed_path: str | Path | None = None,
    sample_size: int = 20,
) -> None:
    reviews = _read_table(reviews_path)
    print("=== Steam Review Dataset Validation ===")
    print(f"reviews path: {reviews_path}")
    print(f"reviews rows: {len(reviews):,}")
    print(f"review columns: {list(reviews.columns)}")
    print()

    if "recommendationid" in reviews.columns:
        duplicate_rec_ids = reviews["recommendationid"].duplicated().sum()
        duplicate_app_rec_ids = reviews.duplicated(subset=["appid", "recommendationid"]).sum()
        print(f"duplicate recommendationid count: {duplicate_rec_ids:,}")
        print(f"duplicate appid+recommendationid count: {duplicate_app_rec_ids:,}")
    else:
        print("duplicate recommendationid count: recommendationid column missing")
    print()

    if "appid" in reviews.columns:
        print("reviews per appid:")
        print(reviews.groupby("appid").size().sort_values(ascending=False).to_string())
    print()

    if "voted_up" in reviews.columns:
        print("voted_up distribution:")
        print(reviews["voted_up"].value_counts(dropna=False).to_string())
    print()

    if "playtime_at_review" in reviews.columns:
        missing_rate = reviews["playtime_at_review"].isna().mean()
        print(f"playtime_at_review missing rate: {missing_rate:.2%}")

    if "review" in reviews.columns:
        review_missing = reviews["review"].isna() | reviews["review"].fillna("").astype(str).str.strip().eq("")
        print(f"review text missing/empty rate: {review_missing.mean():.2%}")
    print()

    if "timestamp_created" in reviews.columns and not reviews.empty:
        created = reviews["timestamp_created"].dropna()
        if not created.empty:
            print("timestamp_created range:")
            print(f"  min: {_format_timestamp(created.min())}")
            print(f"  max: {_format_timestamp(created.max())}")
        else:
            print("timestamp_created range: all missing")
    print()

    if summary_path:
        summary = _read_table(summary_path)
        print(f"summary path: {summary_path}")
        print(f"summary rows: {len(summary):,}")
        if {"appid", "total_reviews_api", "collected_reviews_count"}.issubset(summary.columns):
            compare = summary[["appid", "total_reviews_api", "collected_reviews_count"]].copy()
            compare["collection_ratio"] = compare["collected_reviews_count"] / compare[
                "total_reviews_api"
            ].replace({0: pd.NA})
            print("total_reviews_api vs collected_reviews_count:")
            print(compare.to_string(index=False))
        print()

    if failed_path:
        failed_path = Path(failed_path)
        print(f"failed_appids path: {failed_path}")
        if failed_path.exists():
            failed = _read_table(failed_path)
            if failed.empty:
                print("failed appids: none")
            else:
                print("failed appids:")
                print(failed.to_string(index=False))
        else:
            print("failed appids file does not exist")
        print()

    print(f"sample reviews ({sample_size}):")
    sample_columns = [
        column
        for column in ["appid", "game_name", "recommendationid", "language", "voted_up", "playtime_at_review", "review"]
        if column in reviews.columns
    ]
    if reviews.empty:
        print("no review rows available")
    else:
        sample = reviews[sample_columns].sample(min(sample_size, len(reviews)), random_state=42).copy()
        if "review" in sample.columns:
            sample["review"] = sample["review"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 180)
        print(sample.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Steam review crawler outputs.")
    parser.add_argument("--reviews", default="data/raw/reviews_raw.parquet", help="Path to reviews parquet/csv.")
    parser.add_argument(
        "--summary",
        default="data/raw/review_summary_by_app.parquet",
        help="Path to review summary parquet/csv.",
    )
    parser.add_argument(
        "--failed",
        default="data/logs/failed_appids.csv",
        help="Path to failed_appids.csv.",
    )
    parser.add_argument("--sample-size", type=int, default=20, help="Number of sample reviews to print.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_dataset(args.reviews, args.summary, args.failed, args.sample_size)
