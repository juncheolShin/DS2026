from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import requests
from tqdm.auto import tqdm

from src.utils import (
    SteamForbiddenError,
    SteamRequestError,
    SteamRetryExceeded,
    coerce_float,
    coerce_int,
    get_hash_salt,
    now_utc_iso,
    request_json_with_retry,
    sha256_hash,
)


REVIEW_COLUMNS = [
    "appid",
    "game_name",
    "recommendationid",
    "language",
    "review",
    "is_empty_review",
    "voted_up",
    "timestamp_created",
    "timestamp_updated",
    "steam_purchase",
    "received_for_free",
    "written_during_early_access",
    "votes_up",
    "votes_funny",
    "weighted_vote_score",
    "comment_count",
    "author_steamid_hash",
    "author_num_games_owned",
    "author_num_reviews",
    "playtime_forever",
    "playtime_last_two_weeks",
    "playtime_at_review",
    "last_played",
    "crawled_at",
]

SUMMARY_COLUMNS = [
    "appid",
    "game_name",
    "total_reviews_api",
    "total_positive_api",
    "total_negative_api",
    "review_score",
    "review_score_desc",
    "collected_reviews_count",
    "positive_collected_count",
    "negative_collected_count",
    "first_review_time",
    "last_review_time",
    "crawled_at",
    "termination_reason",
]

FAILED_COLUMNS = [
    "appid",
    "game_name",
    "stage",
    "reason",
    "error",
    "collected_reviews_count",
    "crawled_at",
]


def _review_params(config: dict[str, Any], cursor: str, remaining: int) -> dict[str, Any]:
    crawler_config = config.get("crawler", {})
    requested_page_size = int(crawler_config.get("num_per_page", 100))
    page_size = min(100, requested_page_size, max(1, remaining))
    return {
        "json": 1,
        "filter": crawler_config.get("filter", "recent"),
        "language": crawler_config.get("language", "koreana"),
        "review_type": crawler_config.get("review_type", "all"),
        "purchase_type": crawler_config.get("purchase_type", "all"),
        "num_per_page": page_size,
        "cursor": cursor,
        "filter_offtopic_activity": crawler_config.get("filter_offtopic_activity", 1),
    }


def _normalize_review(
    raw: dict[str, Any],
    *,
    appid: int,
    game_name: str,
    config: dict[str, Any],
    crawled_at: str,
    hash_salt: str,
) -> dict[str, Any] | None:
    recommendationid = raw.get("recommendationid")
    if recommendationid is None:
        return None

    review_text = raw.get("review")
    if review_text is not None:
        review_text = str(review_text)
    is_empty_review = review_text is None or review_text.strip() == ""
    author = raw.get("author") or {}

    return {
        "appid": appid,
        "game_name": game_name,
        "recommendationid": str(recommendationid),
        "language": raw.get("language") or config.get("crawler", {}).get("language", "koreana"),
        "review": review_text,
        "is_empty_review": bool(is_empty_review),
        "voted_up": raw.get("voted_up"),
        "timestamp_created": coerce_int(raw.get("timestamp_created")),
        "timestamp_updated": coerce_int(raw.get("timestamp_updated")),
        "steam_purchase": raw.get("steam_purchase"),
        "received_for_free": raw.get("received_for_free"),
        "written_during_early_access": raw.get("written_during_early_access"),
        "votes_up": coerce_int(raw.get("votes_up")),
        "votes_funny": coerce_int(raw.get("votes_funny")),
        "weighted_vote_score": coerce_float(raw.get("weighted_vote_score")),
        "comment_count": coerce_int(raw.get("comment_count")),
        "author_steamid_hash": sha256_hash(author.get("steamid"), salt=hash_salt),
        "author_num_games_owned": coerce_int(author.get("num_games_owned")),
        "author_num_reviews": coerce_int(author.get("num_reviews")),
        "playtime_forever": coerce_int(author.get("playtime_forever")),
        "playtime_last_two_weeks": coerce_int(author.get("playtime_last_two_weeks")),
        "playtime_at_review": coerce_int(author.get("playtime_at_review")),
        "last_played": coerce_int(author.get("last_played")),
        "crawled_at": crawled_at,
    }


def _summary_from_rows(
    *,
    appid: int,
    game_name: str,
    rows: list[dict[str, Any]],
    query_summary: dict[str, Any],
    crawled_at: str,
    termination_reason: str,
) -> dict[str, Any]:
    timestamps = [row["timestamp_created"] for row in rows if row.get("timestamp_created") is not None]
    voted_up_values = [row.get("voted_up") for row in rows]
    positive_count = sum(value is True for value in voted_up_values)
    negative_count = sum(value is False for value in voted_up_values)

    return {
        "appid": appid,
        "game_name": game_name,
        "total_reviews_api": coerce_int(query_summary.get("total_reviews")),
        "total_positive_api": coerce_int(query_summary.get("total_positive")),
        "total_negative_api": coerce_int(query_summary.get("total_negative")),
        "review_score": coerce_int(query_summary.get("review_score")),
        "review_score_desc": query_summary.get("review_score_desc"),
        "collected_reviews_count": len(rows),
        "positive_collected_count": positive_count,
        "negative_collected_count": negative_count,
        "first_review_time": min(timestamps) if timestamps else None,
        "last_review_time": max(timestamps) if timestamps else None,
        "crawled_at": crawled_at,
        "termination_reason": termination_reason,
    }


def collect_reviews_for_app(
    appid: int,
    game_name: str,
    config: dict[str, Any],
    session: requests.Session,
    logger: logging.Logger | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    crawler_config = config.get("crawler", {})
    endpoint = crawler_config.get("review_endpoint", "https://store.steampowered.com/appreviews/{appid}")
    url = endpoint.format(appid=appid)
    max_reviews = int(crawler_config.get("max_reviews_per_app", 200))
    max_failures = int(crawler_config.get("max_failures", 3))
    sleep_sec = float(crawler_config.get("sleep_sec", 1.0))
    timeout_sec = float(crawler_config.get("timeout_sec", 20))
    retry_config = crawler_config.get("retry", {})
    crawled_at = now_utc_iso()
    hash_salt = get_hash_salt(config)

    cursor = "*"
    seen_cursors: set[str] = set()
    seen_recommendationids: set[str] = set()
    rows: list[dict[str, Any]] = []
    latest_query_summary: dict[str, Any] = {}
    consecutive_failures = 0
    termination_reason = "unknown"
    failure: dict[str, Any] | None = None

    while len(rows) < max_reviews:
        if cursor in seen_cursors:
            termination_reason = "cursor_loop"
            break
        seen_cursors.add(cursor)

        remaining = max_reviews - len(rows)
        params = _review_params(config, cursor, remaining)
        try:
            payload = request_json_with_retry(
                session,
                url,
                params=params,
                timeout_sec=timeout_sec,
                retry_config=retry_config,
                logger=logger,
                context=f"reviews appid={appid}",
            )
            consecutive_failures = 0
        except SteamForbiddenError as exc:
            termination_reason = "http_403"
            failure = {
                "appid": appid,
                "game_name": game_name,
                "stage": "reviews",
                "reason": termination_reason,
                "error": str(exc),
                "collected_reviews_count": len(rows),
                "crawled_at": crawled_at,
            }
            if logger:
                logger.error("review collection stopped for appid=%s: %s", appid, exc)
            break
        except (SteamRetryExceeded, SteamRequestError, requests.RequestException) as exc:
            consecutive_failures += 1
            if logger:
                logger.warning(
                    "review page failed for appid=%s (%s); consecutive failures=%s/%s: %s",
                    appid,
                    game_name,
                    consecutive_failures,
                    max_failures,
                    exc,
                )
            if consecutive_failures > max_failures:
                termination_reason = "max_failures"
                failure = {
                    "appid": appid,
                    "game_name": game_name,
                    "stage": "reviews",
                    "reason": termination_reason,
                    "error": str(exc),
                    "collected_reviews_count": len(rows),
                    "crawled_at": crawled_at,
                }
                break
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            continue

        latest_query_summary = payload.get("query_summary") or latest_query_summary
        page_reviews = payload.get("reviews") or []
        if not page_reviews:
            termination_reason = "no_reviews"
            break

        for raw_review in page_reviews:
            normalized = _normalize_review(
                raw_review,
                appid=appid,
                game_name=game_name,
                config=config,
                crawled_at=crawled_at,
                hash_salt=hash_salt,
            )
            if not normalized:
                continue

            recommendationid = normalized["recommendationid"]
            if recommendationid in seen_recommendationids:
                continue
            seen_recommendationids.add(recommendationid)
            rows.append(normalized)
            if len(rows) >= max_reviews:
                termination_reason = "max_reviews_per_app"
                break

        if len(rows) >= max_reviews:
            break

        next_cursor = payload.get("cursor")
        if not next_cursor:
            termination_reason = "missing_cursor"
            break
        if next_cursor in seen_cursors:
            termination_reason = "cursor_loop"
            break

        cursor = str(next_cursor)
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    final_count = config.get("crawler", {}).get("final_reviews_per_app")
    if final_count is not None and len(rows) > final_count:
        rows.sort(key=lambda x: x.get("weighted_vote_score") or 0.0, reverse=True)
        rows = rows[:final_count]

    summary = _summary_from_rows(
        appid=appid,
        game_name=game_name,
        rows=rows,
        query_summary=latest_query_summary,
        crawled_at=crawled_at,
        termination_reason=termination_reason,
    )
    return rows, summary, failure


def collect_reviews(
    appids_df: pd.DataFrame,
    config: dict[str, Any],
    session: requests.Session,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_reviews: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    iterator = tqdm(appids_df.itertuples(index=False), total=len(appids_df), desc="reviews")
    for item in iterator:
        appid = int(item.appid)
        game_name = str(item.name)
        iterator.set_postfix_str(f"{appid}")
        rows, summary, failure = collect_reviews_for_app(appid, game_name, config, session, logger)
        all_reviews.extend(rows)
        summaries.append(summary)
        if failure:
            failures.append(failure)

        if logger:
            logger.info(
                "appid=%s collected=%s termination=%s",
                appid,
                summary["collected_reviews_count"],
                summary["termination_reason"],
            )

    reviews_df = pd.DataFrame(all_reviews, columns=REVIEW_COLUMNS)
    if not reviews_df.empty:
        reviews_df = reviews_df.drop_duplicates(subset=["appid", "recommendationid"], keep="first")

    summary_df = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
    failed_df = pd.DataFrame(failures, columns=FAILED_COLUMNS)
    return reviews_df, summary_df, failed_df
