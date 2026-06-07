from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from select_strategy_games import (
    candidate_sort_key,
    fetch_review_summary,
    fetch_strategy_search_page,
    parse_search_results,
    rating_bin,
)
from src.utils import load_config, project_path, requests_session


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_TAGS = [
    (19, "Action"),
    (21, "Adventure"),
    (122, "RPG"),
    (9, "Strategy"),
    (599, "Simulation"),
    (597, "Casual"),
    (492, "Indie"),
    (113, "Free To Play"),
    (1664, "Puzzle"),
    (1774, "Shooter"),
    (1667, "Horror"),
    (4106, "Action-Adventure"),
    (1695, "Open World"),
    (1662, "Survival"),
    (3810, "Sandbox"),
    (701, "Sports"),
    (699, "Racing"),
    (1741, "Turn-Based Strategy"),
    (1716, "Roguelike"),
    (1666, "Card Game"),
    (4328, "City Builder"),
]

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Steam tags for balanced overall-rating distributions without modifying appids/raw data."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--target-count", type=int, default=300)
    parser.add_argument("--min-korean-reviews", type=int, default=500)
    parser.add_argument("--max-results", type=int, default=1200)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-check-per-bin", type=int, default=80)
    parser.add_argument("--bin-width", type=int, default=10)
    parser.add_argument("--bin-mode", choices=["percent", "steam_score"], default="percent")
    parser.add_argument("--sort-by-list", default="Reviews_ASC,Reviews_DESC,_ASC")
    parser.add_argument(
        "--tags",
        default="",
        help="Optional comma-separated tag ids to audit. Defaults to common Steam tags.",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--details-output", default="")
    return parser.parse_args()


def requested_tags(args: argparse.Namespace) -> list[tuple[int, str]]:
    if not args.tags.strip():
        return DEFAULT_TAGS
    wanted = {int(value.strip()) for value in args.tags.split(",") if value.strip()}
    known = {tag_id: name for tag_id, name in DEFAULT_TAGS}
    return [(tag_id, known.get(tag_id, f"tag_{tag_id}")) for tag_id in wanted]


def base_selection_config(config: dict[str, Any], args: argparse.Namespace, tag_id: int) -> dict[str, Any]:
    selection = config.get("selection", {})
    return {
        "strategy_tag_id": tag_id,
        "games_category_id": selection.get("games_category_id", 998),
        "search_endpoint": selection.get("search_endpoint", "https://store.steampowered.com/search/results/"),
        "search_page_size": args.page_size,
        "search_timeout_sec": selection.get("search_timeout_sec", 30),
        "review_timeout_sec": selection.get("review_timeout_sec", 20),
        "retry": selection.get("retry", config.get("crawler", {}).get("retry", {})),
    }


def collect_search_candidates(
    session,
    config: dict[str, Any],
    tag_id: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    sel_config = base_selection_config(config, args, tag_id)
    by_appid: dict[int, dict[str, Any]] = {}
    sort_values = [value.strip() for value in args.sort_by_list.split(",") if value.strip()]

    for sort_by in sort_values:
        for start in range(0, args.max_results, args.page_size):
            print(f"tag={tag_id} sort={sort_by} start={start}", flush=True)
            try:
                html, _ = fetch_strategy_search_page(session, config, sel_config, start, sort_by)
            except Exception as exc:
                print(f"  search skipped after error: {exc}")
                break

            rows = parse_search_results(html, first_rank=start + 1)
            if not rows:
                break

            for row in rows:
                if row.get("search_positive_percent") is None:
                    continue
                appid = int(row["appid"])
                row["search_sort"] = sort_by
                previous = by_appid.get(appid)
                if previous is None or candidate_sort_key(row) < candidate_sort_key(previous):
                    by_appid[appid] = row
            time.sleep(0.75)

    return list(by_appid.values())


def eligible_by_bin(
    session,
    config: dict[str, Any],
    tag_id: int,
    tag_name: str,
    search_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows_by_bin: dict[str, list[dict[str, Any]]] = {}
    for row in search_rows:
        if args.bin_mode == "steam_score":
            label = str(row.get("search_review_score_desc") or "Unknown")
        else:
            _, label = rating_bin(float(row["search_positive_percent"]), args.bin_width)
        row = {**row, "overall_rating_bin": label}
        rows_by_bin.setdefault(label, []).append(row)

    eligible_rows: list[dict[str, Any]] = []
    for label in sorted(rows_by_bin):
        rows = sorted(rows_by_bin[label], key=candidate_sort_key)[: args.max_check_per_bin]
        for row in rows:
            appid = int(row["appid"])
            try:
                summary = fetch_review_summary(
                    session,
                    config,
                    appid,
                    language="koreana",
                    timeout_sec=float(config.get("selection", {}).get("review_timeout_sec", 20)),
                )
            except Exception as exc:
                print(f"  Korean review check failed appid={appid}: {exc}")
                time.sleep(0.2)
                continue

            korean_reviews = int(summary.get("total_reviews") or 0)
            if korean_reviews >= args.min_korean_reviews:
                korean_positive = int(summary.get("total_positive") or 0)
                eligible_rows.append(
                    {
                        "tag_id": tag_id,
                        "tag_name": tag_name,
                        "appid": appid,
                        "name": row["name"],
                        "overall_rating_bin": label,
                        "steam_score_desc": row.get("search_review_score_desc"),
                        "steam_score_desc_ko": STEAM_SCORE_LABEL_KO.get(str(row.get("search_review_score_desc") or "")),
                        "overall_positive_review_percent": float(row["search_positive_percent"]),
                        "search_total_reviews": row.get("search_total_reviews"),
                        "korean_reviews": korean_reviews,
                        "korean_positive_percent": round(korean_positive / korean_reviews * 100, 2)
                        if korean_reviews
                        else None,
                        "search_rank": row.get("search_rank"),
                        "search_sort": row.get("search_sort"),
                    }
                )
            time.sleep(0.2)

    return pd.DataFrame(eligible_rows)


def summarize_tag(
    tag_id: int,
    tag_name: str,
    eligible: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.bin_mode == "steam_score":
        bin_labels = STEAM_SCORE_LABELS
        required_label = "required_per_score_bins"
        possible_label = "can_make_300_equal_score_bins"
        max_equal_all_label = "max_equal_count_all_score_bins"
    else:
        bin_labels = [f"{start:02d}-{start + args.bin_width:03d}" for start in range(0, 100, args.bin_width)]
        required_label = "required_per_10_bins"
        possible_label = "can_make_300_equal_10_bins"
        max_equal_all_label = "max_equal_count_all_10_bins"

    required_per_bin = args.target_count // len(bin_labels)
    counts = eligible["overall_rating_bin"].value_counts().to_dict() if not eligible.empty else {}
    active_counts = [counts.get(label, 0) for label in bin_labels if counts.get(label, 0) > 0]
    active_bins = len(active_counts)
    max_equal_all_10_bins = min(counts.get(label, 0) for label in bin_labels) * len(bin_labels)
    max_equal_active_bins = min(active_counts) * active_bins if active_counts else 0
    can_target_all_bins = all(counts.get(label, 0) >= required_per_bin for label in bin_labels)

    row = {
        "tag_id": tag_id,
        "tag_name": tag_name,
        "eligible_count": int(len(eligible)),
        "active_bins": active_bins,
        required_label: required_per_bin,
        possible_label: bool(can_target_all_bins),
        max_equal_all_label: int(max_equal_all_10_bins),
        "max_equal_count_active_bins": int(max_equal_active_bins),
    }
    for label in bin_labels:
        safe_label = label.replace(" ", "_").replace("-", "_")
        row[f"bin_{safe_label}"] = int(counts.get(label, 0))
    return row


def output_paths(args: argparse.Namespace) -> tuple[str, str]:
    if args.output:
        output = args.output
    elif args.bin_mode == "steam_score":
        output = "data/logs/category_score_balance_audit.csv"
    else:
        output = "data/logs/category_balance_audit.csv"

    if args.details_output:
        details_output = args.details_output
    elif args.bin_mode == "steam_score":
        details_output = "data/logs/category_score_balance_audit_details.csv"
    else:
        details_output = "data/logs/category_balance_audit_details.csv"
    return output, details_output


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []

    with requests_session(config) as session:
        for tag_id, tag_name in requested_tags(args):
            print(f"=== Auditing {tag_name} ({tag_id}) ===", flush=True)
            search_rows = collect_search_candidates(session, config, tag_id, args)
            eligible = eligible_by_bin(session, config, tag_id, tag_name, search_rows, args)
            detail_frames.append(eligible)
            summary = summarize_tag(tag_id, tag_name, eligible, args)
            summary_rows.append(summary)
            max_equal_key = (
                "max_equal_count_all_score_bins"
                if args.bin_mode == "steam_score"
                else "max_equal_count_all_10_bins"
            )
            print(
                f"{tag_name}: eligible={summary['eligible_count']} "
                f"active_bins={summary['active_bins']} "
                f"max_equal_all={summary[max_equal_key]} "
                f"max_equal_active={summary['max_equal_count_active_bins']}"
            )

    summary_df = pd.DataFrame(summary_rows)
    details_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()

    output_arg, details_output_arg = output_paths(args)
    output = project_path(config, output_arg)
    details_output = project_path(config, details_output_arg)
    output.parent.mkdir(parents=True, exist_ok=True)
    details_output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output, index=False, encoding="utf-8-sig")
    details_df.to_csv(details_output, index=False, encoding="utf-8-sig")

    print("=== Audit Complete ===")
    sort_column = "max_equal_count_all_score_bins" if args.bin_mode == "steam_score" else "max_equal_count_all_10_bins"
    print(summary_df.sort_values([sort_column, "eligible_count"], ascending=False).to_string(index=False))
    print(f"Saved summary: {output}")
    print(f"Saved details: {details_output}")


if __name__ == "__main__":
    main()
