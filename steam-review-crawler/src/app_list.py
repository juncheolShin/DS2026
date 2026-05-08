from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

from src.utils import project_path, request_json_with_retry


REQUIRED_APPID_COLUMNS = {"appid", "name"}


def read_target_appids(config: dict[str, Any]) -> pd.DataFrame:
    appids_path = project_path(config, config["paths"]["appids_csv"])
    if not appids_path.exists():
        raise FileNotFoundError(f"appid input file not found: {appids_path}")

    df = pd.read_csv(appids_path)
    missing = REQUIRED_APPID_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"appids.csv is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["appid"] = pd.to_numeric(df["appid"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["appid"]).copy()
    df["appid"] = df["appid"].astype(int)
    df["name"] = df["name"].fillna("").astype(str)

    if "priority" in df.columns:
        df["_priority_sort"] = pd.to_numeric(df["priority"], errors="coerce")
        df = df.sort_values("_priority_sort", na_position="last").drop(columns="_priority_sort")

    df = df.drop_duplicates(subset=["appid"], keep="first").reset_index(drop=True)
    return df


def fetch_steam_app_list(
    config: dict[str, Any],
    session: requests.Session,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Fetch the public Steam app list for future expansion beyond appids.csv."""
    app_list_config = config.get("app_list", {})
    endpoint = app_list_config.get(
        "endpoint", "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
    )
    payload = request_json_with_retry(
        session,
        endpoint,
        timeout_sec=float(app_list_config.get("timeout_sec", 30)),
        retry_config=app_list_config.get("retry", {}),
        logger=logger,
        context="steam app list",
    )
    apps = payload.get("applist", {}).get("apps", [])
    return pd.DataFrame(apps)
