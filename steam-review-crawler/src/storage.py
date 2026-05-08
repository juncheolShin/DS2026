from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.metadata_crawler import METADATA_COLUMNS
from src.review_crawler import FAILED_COLUMNS, REVIEW_COLUMNS, SUMMARY_COLUMNS
from src.utils import project_path


def _ordered_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    extras = [column for column in result.columns if column not in columns]
    return result[columns + extras]


def _save_table(
    df: pd.DataFrame,
    *,
    parquet_path: Path,
    csv_path: Path,
    save_parquet: bool,
    save_csv: bool,
    logger: logging.Logger | None = None,
) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if save_parquet:
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        if logger:
            logger.info("saved parquet: rows=%s path=%s", len(df), parquet_path)
    if save_csv:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        if logger:
            logger.info("saved csv: rows=%s path=%s", len(df), csv_path)


def save_outputs(
    *,
    reviews_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    failed_df: pd.DataFrame,
    config: dict[str, Any],
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    output = config.get("output", {})
    save_csv = bool(output.get("save_csv", True))
    save_parquet = bool(output.get("save_parquet", True))
    paths = config.get("paths", {})

    reviews_df = _ordered_frame(reviews_df, REVIEW_COLUMNS)
    summary_df = _ordered_frame(summary_df, SUMMARY_COLUMNS)
    metadata_df = _ordered_frame(metadata_df, METADATA_COLUMNS)
    failed_df = _ordered_frame(failed_df, FAILED_COLUMNS)

    reviews_parquet = project_path(config, paths.get("reviews_raw_parquet", "data/raw/reviews_raw.parquet"))
    summary_parquet = project_path(config, paths.get("review_summary_parquet", "data/raw/review_summary_by_app.parquet"))
    metadata_parquet = project_path(config, paths.get("app_metadata_parquet", "data/raw/app_metadata.parquet"))
    failed_csv = project_path(config, paths.get("failed_appids_csv", "data/logs/failed_appids.csv"))

    _save_table(
        reviews_df,
        parquet_path=reviews_parquet,
        csv_path=reviews_parquet.with_suffix(".csv"),
        save_parquet=save_parquet,
        save_csv=save_csv,
        logger=logger,
    )
    _save_table(
        summary_df,
        parquet_path=summary_parquet,
        csv_path=summary_parquet.with_suffix(".csv"),
        save_parquet=save_parquet,
        save_csv=save_csv,
        logger=logger,
    )
    _save_table(
        metadata_df,
        parquet_path=metadata_parquet,
        csv_path=metadata_parquet.with_suffix(".csv"),
        save_parquet=save_parquet,
        save_csv=save_csv,
        logger=logger,
    )

    failed_csv.parent.mkdir(parents=True, exist_ok=True)
    failed_df.to_csv(failed_csv, index=False, encoding="utf-8-sig")
    if logger:
        logger.info("saved failed appids csv: rows=%s path=%s", len(failed_df), failed_csv)

    return {
        "reviews_parquet": reviews_parquet,
        "summary_parquet": summary_parquet,
        "metadata_parquet": metadata_parquet,
        "failed_appids_csv": failed_csv,
    }
