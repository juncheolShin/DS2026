from __future__ import annotations

import argparse

from src.app_list import read_target_appids
from src.metadata_crawler import collect_app_metadata
from src.review_crawler import collect_reviews
from src.storage import save_outputs
from src.utils import ensure_directories, load_config, requests_session, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Steam reviews and metadata for ABSA analysis.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--language", choices=["koreana", "all"], help="Override Steam review language.")
    parser.add_argument("--max-reviews-per-app", type=int, help="Override maximum reviews collected per app.")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Limit each app to config test_mode.max_reviews_per_app, default 200.",
    )
    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    config.setdefault("crawler", {})
    if args.test_mode:
        config["crawler"]["max_reviews_per_app"] = int(config.get("test_mode", {}).get("max_reviews_per_app", 200))
    if args.language:
        config["crawler"]["language"] = args.language
    if args.max_reviews_per_app is not None:
        config["crawler"]["max_reviews_per_app"] = args.max_reviews_per_app
    return config


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    ensure_directories(config)
    logger = setup_logger(config)

    logger.info("loading target appids")
    appids_df = read_target_appids(config)
    logger.info("loaded %s target appids", len(appids_df))
    logger.info(
        "review language=%s filter=%s max_reviews_per_app=%s",
        config.get("crawler", {}).get("language"),
        config.get("crawler", {}).get("filter"),
        config.get("crawler", {}).get("max_reviews_per_app"),
    )

    with requests_session(config) as session:
        metadata_df = collect_app_metadata(appids_df, config, session, logger)
        reviews_df, summary_df, failed_df = collect_reviews(appids_df, config, session, logger)

    outputs = save_outputs(
        reviews_df=reviews_df,
        summary_df=summary_df,
        metadata_df=metadata_df,
        failed_df=failed_df,
        config=config,
        logger=logger,
    )

    logger.info("collection complete")
    for name, path in outputs.items():
        logger.info("%s: %s", name, path)


if __name__ == "__main__":
    main()
