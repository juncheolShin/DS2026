from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import requests
from tqdm.auto import tqdm

from src.utils import coerce_int, json_dumps, now_utc_iso, request_json_with_retry


METADATA_COLUMNS = [
    "appid",
    "name",
    "release_date",
    "developers",
    "publishers",
    "genres",
    "categories",
    "price_overview",
    "metacritic_score",
    "is_free",
    "required_age",
    "short_description",
    "header_image",
    "metadata_source",
    "crawled_at",
]


def _descriptions(items: Any) -> list[str] | None:
    if not isinstance(items, list):
        return None
    values = []
    for item in items:
        if isinstance(item, dict) and item.get("description"):
            values.append(str(item["description"]))
        elif isinstance(item, str):
            values.append(item)
    return values


def _failed_metadata_row(appid: int, name: str, crawled_at: str) -> dict[str, Any]:
    return {
        "appid": appid,
        "name": name,
        "release_date": None,
        "developers": None,
        "publishers": None,
        "genres": None,
        "categories": None,
        "price_overview": None,
        "metacritic_score": None,
        "is_free": None,
        "required_age": None,
        "short_description": None,
        "header_image": None,
        "metadata_source": "failed",
        "crawled_at": crawled_at,
    }


def _metadata_row_from_payload(appid: int, fallback_name: str, data: dict[str, Any], crawled_at: str) -> dict[str, Any]:
    release_date = data.get("release_date") or {}
    metacritic = data.get("metacritic") or {}
    name = data.get("name") or fallback_name
    return {
        "appid": appid,
        "name": name,
        "release_date": release_date.get("date"),
        "developers": json_dumps(data.get("developers")),
        "publishers": json_dumps(data.get("publishers")),
        "genres": json_dumps(_descriptions(data.get("genres"))),
        "categories": json_dumps(_descriptions(data.get("categories"))),
        "price_overview": json_dumps(data.get("price_overview")),
        "metacritic_score": coerce_int(metacritic.get("score")),
        "is_free": data.get("is_free"),
        "required_age": coerce_int(data.get("required_age")),
        "short_description": data.get("short_description"),
        "header_image": data.get("header_image"),
        "metadata_source": "appdetails",
        "crawled_at": crawled_at,
    }


def collect_app_metadata(
    appids_df: pd.DataFrame,
    config: dict[str, Any],
    session: requests.Session,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    metadata_config = config.get("metadata", {})
    crawled_at = now_utc_iso()

    if not metadata_config.get("enabled", True):
        rows = []
        for item in appids_df.itertuples(index=False):
            row = _failed_metadata_row(int(item.appid), str(item.name), crawled_at)
            row["metadata_source"] = "disabled"
            rows.append(row)
        return pd.DataFrame(rows, columns=METADATA_COLUMNS)

    endpoint = metadata_config.get("appdetails_endpoint", "https://store.steampowered.com/api/appdetails")
    rows: list[dict[str, Any]] = []

    iterator = tqdm(appids_df.itertuples(index=False), total=len(appids_df), desc="metadata")
    for item in iterator:
        appid = int(item.appid)
        name = str(item.name)
        try:
            payload = request_json_with_retry(
                session,
                endpoint,
                params={
                    "appids": appid,
                    "cc": metadata_config.get("cc", "kr"),
                    "l": metadata_config.get("language", "korean"),
                },
                timeout_sec=float(metadata_config.get("timeout_sec", 20)),
                retry_config=metadata_config.get("retry", {}),
                logger=logger,
                context=f"appdetails appid={appid}",
            )
            app_payload = payload.get(str(appid), {})
            if not app_payload.get("success"):
                raise ValueError(f"appdetails success=false for appid={appid}")
            data = app_payload.get("data") or {}
            rows.append(_metadata_row_from_payload(appid, name, data, crawled_at))
        except Exception as exc:  # Metadata should never block review collection.
            if logger:
                logger.warning("metadata failed for appid=%s (%s): %s", appid, name, exc)
            rows.append(_failed_metadata_row(appid, name, crawled_at))

        sleep_sec = float(metadata_config.get("sleep_sec", 0.5))
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    return pd.DataFrame(rows, columns=METADATA_COLUMNS)
