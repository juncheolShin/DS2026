from __future__ import annotations

import argparse
import html
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.utils import load_config, project_path, request_json_with_retry, requests_session


STEAM_SCORE_LABELS = [
    "Overwhelmingly Negative",
    "Very Negative",
    "Negative",
    "Mostly Negative",
    "Mixed",
    "Mostly Positive",
    "Positive",
    "Very Positive",
    "Overwhelmingly Positive",
]

STEAM_SCORE_LABEL_KO = {
    "Overwhelmingly Negative": "압도적으로 부정적",
    "Very Negative": "매우 부정적",
    "Negative": "부정적",
    "Mostly Negative": "대체로 부정적",
    "Mixed": "복합적",
    "Mostly Positive": "대체로 긍정적",
    "Positive": "긍정적",
    "Very Positive": "매우 긍정적",
    "Overwhelmingly Positive": "압도적으로 긍정적",
}


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


SEARCH_ROW_RE = re.compile(
    r'<a\s+[^>]*href="https://store\.steampowered\.com/app/(\d+)/[^"]*"[^>]*'
    r'class="[^"]*search_result_row[^"]*"[^>]*>.*?</a>',
    re.DOTALL,
)
TITLE_RE = re.compile(r'<span class="title">(.*?)</span>', re.DOTALL)
TOOLTIP_RE = re.compile(r'data-tooltip-html="([^"]+)"', re.DOTALL)
REVIEW_COUNT_RE = re.compile(r"of\s+the\s+([\d,]+)\s+(?:user\s+)?reviews", re.IGNORECASE)
POSITIVE_PERCENT_RE = re.compile(r"(\d{1,3})%\s+of\s+the\s+[\d,]+\s+(?:user\s+)?reviews", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select Steam Strategy games with a balanced distribution of overall positive review ratios."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--target-count", type=int, help="Override selection.target_count.")
    parser.add_argument("--min-korean-reviews", type=int, help="Override selection.min_korean_reviews.")
    parser.add_argument("--max-results", type=int, help="Override selection.max_search_results.")
    return parser.parse_args()


def selection_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    defaults = {
        "target_count": 300,
        "strategy_tag_id": None,
        "games_category_id": 998,
        "max_search_results": 3000,
        "search_page_size": 100,
        "sort_by": "Reviews_DESC",
        "sort_by_list": ["Reviews_ASC", "Reviews_DESC", "_ASC"],
        "min_global_reviews": 300,
        "min_korean_reviews": 500,
        "bin_mode": "steam_score",
        "rating_bin_width": 10,
        "max_candidates_per_bin_check": 600,
        "allow_bin_redistribution": True,
        "quota_over_active_bins": True,
        "require_target_count": True,
        "sleep_sec": 0.15,
        "search_sleep_sec": 1.5,
        "rate_limit_sleep_sec": 45.0,
        "search_timeout_sec": 30,
        "review_timeout_sec": 20,
        "output_appids_csv": "data/input/appids.csv",
        "candidates_csv": "data/logs/strategy_candidates.csv",
        "selected_csv": "data/logs/strategy_selected_games.csv",
        "backup_existing_appids": True,
    }
    merged = {**defaults, **config.get("selection", {})}
    if args.target_count is not None:
        merged["target_count"] = args.target_count
    if args.min_korean_reviews is not None:
        merged["min_korean_reviews"] = args.min_korean_reviews
    if args.max_results is not None:
        merged["max_search_results"] = args.max_results
    return merged


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return html.unescape(TAG_RE.sub(" ", value)).strip()


def parse_review_tooltip(raw_tooltip: str) -> dict[str, Any]:
    tooltip = html.unescape(raw_tooltip or "")
    tooltip_text = clean_text(tooltip.replace("<br>", " "))
    first_line = clean_text(tooltip.split("<br>")[0]) if "<br>" in tooltip else ""

    count_match = REVIEW_COUNT_RE.search(tooltip_text)
    percent_match = POSITIVE_PERCENT_RE.search(tooltip_text)
    return {
        "search_review_score_desc": first_line or None,
        "search_total_reviews": int(count_match.group(1).replace(",", "")) if count_match else None,
        "search_positive_percent": float(percent_match.group(1)) if percent_match else None,
    }


def fetch_strategy_search_page(
    session: requests.Session,
    config: dict[str, Any],
    sel_config: dict[str, Any],
    start: int,
    sort_by: str,
) -> tuple[str, int | None]:
    params = {
        "start": start,
        "count": int(sel_config["search_page_size"]),
        "sort_by": sort_by,
        "category1": int(sel_config["games_category_id"]),
        "cc": "kr",
        "l": "english",
        "infinite": 1,
    }
    tag_id = sel_config.get("strategy_tag_id")
    if tag_id not in (None, "", "null"):
        params["tags"] = int(tag_id)

    payload = request_json_with_retry(
        session,
        sel_config.get("search_endpoint", "https://store.steampowered.com/search/results/"),
        params=params,
        timeout_sec=float(sel_config["search_timeout_sec"]),
        retry_config=sel_config.get("retry", config.get("crawler", {}).get("retry", {})),
        context=f"strategy search sort={sort_by} start={start}",
    )
    return str(payload.get("results_html", "")), payload.get("total_count")


def parse_search_results(results_html: str, first_rank: int) -> list[dict[str, Any]]:
    candidates = []
    for offset, match in enumerate(SEARCH_ROW_RE.finditer(results_html)):
        block = match.group(0)
        appid = int(match.group(1))
        title_match = TITLE_RE.search(block)
        tooltip_match = TOOLTIP_RE.search(block)
        tooltip_info = parse_review_tooltip(tooltip_match.group(1) if tooltip_match else "")
        candidates.append(
            {
                "appid": appid,
                "name": clean_text(title_match.group(1) if title_match else "unknown"),
                "search_rank": first_rank + offset,
                **tooltip_info,
            }
        )
    return candidates


def candidate_sort_key(row: dict[str, Any]) -> tuple[int, int]:
    return (-(int(row.get("search_total_reviews") or 0)), int(row.get("search_rank") or 10**9))


def fetch_review_summary(
    session: requests.Session,
    config: dict[str, Any],
    appid: int,
    language: str,
    timeout_sec: float,
) -> dict[str, Any]:
    endpoint = config.get("crawler", {}).get("review_endpoint", "https://store.steampowered.com/appreviews/{appid}")
    payload = request_json_with_retry(
        session,
        endpoint.format(appid=appid),
        params={
            "json": 1,
            "filter": "all",
            "language": language,
            "review_type": "all",
            "purchase_type": "all",
            "num_per_page": 0,
            "cursor": "*",
            "filter_offtopic_activity": 1,
        },
        timeout_sec=timeout_sec,
        retry_config=config.get("overall_rating", {}).get("retry", config.get("crawler", {}).get("retry", {})),
        context=f"review summary appid={appid} language={language}",
    )
    return payload.get("query_summary") or {}


def rating_bin(percent: float, width: int) -> tuple[int, str]:
    start = min(100 - width, int(percent // width) * width)
    end = start + width
    return start, f"{start:02d}-{end:03d}"


def collect_candidates(
    config: dict[str, Any],
    sel_config: dict[str, Any],
    session: requests.Session,
) -> pd.DataFrame:
    max_results = int(sel_config["max_search_results"])
    page_size = int(sel_config["search_page_size"])
    min_global_reviews = int(sel_config["min_global_reviews"])
    min_korean_reviews = int(sel_config["min_korean_reviews"])
    sleep_sec = float(sel_config["sleep_sec"])

    search_rows: list[dict[str, Any]] = []
    seen_appids: set[int] = set()
    total_count: int | None = None

    sort_values = sel_config.get("sort_by_list") or [sel_config.get("sort_by", "Reviews_DESC")]
    if isinstance(sort_values, str):
        sort_values = [sort_values]

    scope = "Strategy-tagged games" if sel_config.get("strategy_tag_id") not in (None, "", "null") else "all Steam games"
    print(f"=== Fetching {scope} from Steam Store search ===")
    by_appid: dict[int, dict[str, Any]] = {}
    for sort_by in sort_values:
        total_count = None
        for start in range(0, max_results, page_size):
            print(f"search sort={sort_by} start={start}...", flush=True)
            for attempt in range(1, 3):
                try:
                    results_html, page_total = fetch_strategy_search_page(
                        session,
                        config,
                        sel_config,
                        start,
                        str(sort_by),
                    )
                    break
                except Exception as exc:
                    if attempt >= 2:
                        print(f"Search failed for sort={sort_by} start={start}: {exc}")
                        results_html, page_total = "", total_count
                        break
                    sleep_for = float(sel_config.get("rate_limit_sleep_sec", 45.0))
                    print(f"Search throttled/failed; sleeping {sleep_for:.1f}s before retry: {exc}")
                    time.sleep(sleep_for)
            if page_total is not None:
                total_count = int(page_total)
            page_candidates = parse_search_results(results_html, first_rank=start + 1)
            if not page_candidates:
                print(f"No more search results for sort={sort_by}.")
                break

            for candidate in page_candidates:
                candidate["search_sort"] = sort_by
                appid = int(candidate["appid"])
                previous = by_appid.get(appid)
                if previous is None or candidate_sort_key(candidate) < candidate_sort_key(previous):
                    by_appid[appid] = candidate
                    seen_appids.add(appid)

            if total_count is not None and start + page_size >= total_count:
                break
            time.sleep(float(sel_config.get("search_sleep_sec", 1.5)))

    search_rows = list(by_appid.values())

    print(f"Fetched unique search candidates: {len(search_rows)}")
    pre_filtered = []
    for row in search_rows:
        if (row.get("search_total_reviews") or 0) < min_global_reviews:
            continue
        if row.get("search_positive_percent") is None:
            continue
        if sel_config.get("bin_mode") == "steam_score":
            bin_label = str(row.get("search_review_score_desc") or "Unknown")
            bin_start = STEAM_SCORE_LABELS.index(bin_label) if bin_label in STEAM_SCORE_LABELS else len(STEAM_SCORE_LABELS)
        else:
            bin_start, bin_label = rating_bin(float(row["search_positive_percent"]), int(sel_config["rating_bin_width"]))
        pre_filtered.append({**row, "overall_rating_bin_start": bin_start, "overall_rating_bin": bin_label})
    print(f"Candidates with >= {min_global_reviews} global reviews: {len(pre_filtered)}")

    if sel_config.get("bin_mode") == "steam_score":
        bin_labels = STEAM_SCORE_LABELS + ["Unknown"]
    else:
        bin_width = int(sel_config["rating_bin_width"])
        bin_labels = [f"{start:02d}-{start + bin_width:03d}" for start in range(0, 100, bin_width)]
    grouped_for_check: dict[str, list[dict[str, Any]]] = {label: [] for label in bin_labels}
    for row in pre_filtered:
        grouped_for_check[str(row["overall_rating_bin"])].append(row)

    print("Search candidate distribution before Korean filter:")
    for label in bin_labels:
        rows = sorted(grouped_for_check[label], key=candidate_sort_key)
        grouped_for_check[label] = rows
        print(f"  {label}: {len(rows)}")

    max_check_per_bin = sel_config.get("max_candidates_per_bin_check")
    rows_to_check: list[dict[str, Any]] = []
    for label in bin_labels:
        rows = grouped_for_check[label]
        if max_check_per_bin is not None:
            rows = rows[: int(max_check_per_bin)]
        rows_to_check.extend(rows)

    rows_to_check = sorted(rows_to_check, key=lambda item: (item["overall_rating_bin_start"], *candidate_sort_key(item)))
    print(f"Candidates queued for Korean review check: {len(rows_to_check)}")

    valid_rows: list[dict[str, Any]] = []
    print("=== Checking Korean review availability ===")
    for index, row in enumerate(rows_to_check, start=1):
        appid = int(row["appid"])
        name = row["name"]
        print(f"[{index}/{len(rows_to_check)}] {appid} {name}...", flush=True)
        try:
            korean_summary = fetch_review_summary(
                session,
                config,
                appid,
                language="koreana",
                timeout_sec=float(sel_config["review_timeout_sec"]),
            )
            korean_reviews = int(korean_summary.get("total_reviews") or 0)
            korean_positive = int(korean_summary.get("total_positive") or 0)
            if korean_reviews < min_korean_reviews:
                time.sleep(sleep_sec)
                continue

            overall_percent = row.get("search_positive_percent")
            overall_summary: dict[str, Any] = {}
            if overall_percent is None:
                overall_summary = fetch_review_summary(
                    session,
                    config,
                    appid,
                    language="all",
                    timeout_sec=float(sel_config["review_timeout_sec"]),
                )
                total_reviews = int(overall_summary.get("total_reviews") or 0)
                total_positive = int(overall_summary.get("total_positive") or 0)
                overall_percent = (total_positive / total_reviews * 100) if total_reviews else None

            if overall_percent is None:
                time.sleep(sleep_sec)
                continue

            valid_rows.append(
                {
                    **row,
                    "korean_reviews": korean_reviews,
                    "korean_positive": korean_positive,
                    "korean_positive_percent": round(korean_positive / korean_reviews * 100, 2)
                    if korean_reviews
                    else None,
                    "steam_score_desc": row.get("search_review_score_desc"),
                    "steam_score_desc_ko": STEAM_SCORE_LABEL_KO.get(str(row.get("search_review_score_desc") or "")),
                    "overall_positive_review_percent": round(float(overall_percent), 2),
                    "review_score": overall_summary.get("review_score"),
                    "review_score_desc": overall_summary.get("review_score_desc")
                    or row.get("search_review_score_desc"),
                }
            )
        except Exception as exc:
            print(f"  skipped due to error: {exc}")
        time.sleep(sleep_sec)

    return pd.DataFrame(valid_rows)


def balanced_queue_select(
    candidates: pd.DataFrame,
    target_count: int,
    bin_width: int,
    *,
    allow_redistribution: bool,
    quota_over_active_bins: bool,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    candidates = candidates.sort_values(
        ["overall_rating_bin_start", "search_total_reviews", "search_rank"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    if "steam_score_desc" in candidates.columns and candidates["steam_score_desc"].notna().any():
        bin_labels = [label for label in STEAM_SCORE_LABELS if label in set(candidates["overall_rating_bin"])]
        unknown_count = (candidates["overall_rating_bin"] == "Unknown").sum()
        if unknown_count:
            bin_labels.append("Unknown")
    else:
        bin_labels = [f"{start:02d}-{start + bin_width:03d}" for start in range(0, 100, bin_width)]
    queues: dict[str, list[int]] = defaultdict(list)
    for idx, row in candidates.iterrows():
        queues[str(row["overall_rating_bin"])].append(idx)

    quota_labels = [label for label in bin_labels if queues[label]] if quota_over_active_bins else bin_labels
    quota = target_count // len(quota_labels)
    remainder = target_count % len(quota_labels)
    selected_indices: list[int] = []
    selected_by_bin = {label: 0 for label in bin_labels}
    offsets = {label: 0 for label in bin_labels}

    for i, label in enumerate(quota_labels):
        bin_quota = quota + (1 if i < remainder else 0)
        take = min(bin_quota, len(queues[label]))
        selected_indices.extend(queues[label][:take])
        offsets[label] = take
        selected_by_bin[label] = take

    while allow_redistribution and len(selected_indices) < target_count:
        available = [label for label in bin_labels if offsets[label] < len(queues[label])]
        if not available:
            break
        label = min(available, key=lambda item: (selected_by_bin[item], item))
        selected_indices.append(queues[label][offsets[label]])
        offsets[label] += 1
        selected_by_bin[label] += 1

    selected = candidates.loc[selected_indices].copy()
    selected["selection_order"] = range(1, len(selected) + 1)
    selected = selected.sort_values(["overall_rating_bin_start", "search_rank"]).reset_index(drop=True)
    selected["priority"] = range(1, len(selected) + 1)
    return selected


def backup_existing_appids(config: dict[str, Any], output_path: Path) -> None:
    if not output_path.exists():
        return
    logs_dir = project_path(config, config.get("paths", {}).get("logs_dir", "data/logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = logs_dir / f"appids_backup_{timestamp}.csv"
    backup_path.write_bytes(output_path.read_bytes())
    print(f"Backed up existing appids.csv -> {backup_path}")


def save_outputs(config: dict[str, Any], sel_config: dict[str, Any], candidates: pd.DataFrame, selected: pd.DataFrame) -> bool:
    candidates_path = project_path(config, sel_config["candidates_csv"])
    selected_path = project_path(config, sel_config["selected_csv"])
    appids_path = project_path(config, sel_config["output_appids_csv"])

    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    appids_path.parent.mkdir(parents=True, exist_ok=True)

    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    selected.to_csv(selected_path, index=False, encoding="utf-8-sig")

    target_count = int(sel_config["target_count"])
    require_target = bool(sel_config.get("require_target_count", True))
    if require_target and len(selected) < target_count:
        print(
            f"Not overwriting appids.csv because selected {len(selected)} < required target_count {target_count}."
        )
        print(f"Inspect partial selection at: {selected_path}")
        return False

    if bool(sel_config.get("backup_existing_appids", True)):
        backup_existing_appids(config, appids_path)

    appids_columns = [
        "appid",
        "name",
        "overall_rating_bin",
        "steam_score_desc",
        "steam_score_desc_ko",
        "overall_positive_review_percent",
        "korean_reviews",
        "korean_positive_percent",
        "search_rank",
        "review_score_desc",
        "priority",
    ]
    selected[appids_columns].to_csv(appids_path, index=False, encoding="utf-8-sig")
    print(f"Saved candidates: {candidates_path}")
    print(f"Saved selected games: {selected_path}")
    print(f"Saved appids.csv: {appids_path}")
    return True


def print_distribution(selected: pd.DataFrame) -> None:
    print("=== Selected rating-bin distribution ===")
    if selected.empty:
        print("No selected games.")
        return
    counts = selected["overall_rating_bin"].value_counts().sort_index()
    print(counts.to_string())
    print()
    print("Lowest rating examples:")
    print(selected.head(5)[["appid", "name", "overall_positive_review_percent", "korean_reviews"]].to_string(index=False))
    print()
    print("Highest rating examples:")
    print(selected.tail(5)[["appid", "name", "overall_positive_review_percent", "korean_reviews"]].to_string(index=False))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    sel_config = selection_config(config, args)

    with requests_session(config) as session:
        candidates = collect_candidates(config, sel_config, session)

    target_count = int(sel_config["target_count"])
    selected = balanced_queue_select(
        candidates,
        target_count=target_count,
        bin_width=int(sel_config["rating_bin_width"]),
        allow_redistribution=bool(sel_config.get("allow_bin_redistribution", False)),
        quota_over_active_bins=bool(sel_config.get("quota_over_active_bins", True)),
    )

    if len(selected) < target_count:
        print(f"Warning: selected only {len(selected)} games out of requested {target_count}.")

    wrote_appids = save_outputs(config, sel_config, candidates, selected)
    print_distribution(selected)
    if not wrote_appids:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
