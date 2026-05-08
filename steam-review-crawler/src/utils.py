from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests
import yaml
from dotenv import load_dotenv


class SteamRequestError(Exception):
    """Base error for Steam request failures."""


class SteamForbiddenError(SteamRequestError):
    """Raised when Steam returns HTTP 403."""


class SteamRetryExceeded(SteamRequestError):
    """Raised when retry attempts are exhausted."""


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    base_dir = path.parent
    env_file = config.get("env_file")
    if env_file:
        load_dotenv(base_dir / env_file)

    config["_base_dir"] = str(base_dir)
    return config


def project_path(config: Mapping[str, Any], path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(str(config["_base_dir"])) / path


def ensure_directories(config: Mapping[str, Any]) -> None:
    paths = config.get("paths", {})
    for key in ("raw_dir", "logs_dir"):
        if paths.get(key):
            project_path(config, paths[key]).mkdir(parents=True, exist_ok=True)

    appids_path = paths.get("appids_csv")
    if appids_path:
        project_path(config, appids_path).parent.mkdir(parents=True, exist_ok=True)


def setup_logger(config: Mapping[str, Any]) -> logging.Logger:
    ensure_directories(config)
    logger = logging.getLogger("steam_review_crawler")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = project_path(config, config.get("paths", {}).get("logs_dir", "data/logs"))
    file_handler = logging.FileHandler(log_dir / "crawler.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def requests_session(config: Mapping[str, Any]) -> requests.Session:
    session = requests.Session()
    user_agent = (
        config.get("crawler", {}).get("user_agent")
        or "steam-review-crawler/0.1 (academic data science project)"
    )
    session.headers.update({"User-Agent": user_agent})
    return session


def get_hash_salt(config: Mapping[str, Any]) -> str:
    output = config.get("output", {})
    env_name = output.get("privacy_hash_salt_env", "STEAM_REVIEW_HASH_SALT")
    return os.getenv(env_name, output.get("hash_salt", ""))


def sha256_hash(value: Any, salt: str = "") -> str | None:
    if value is None:
        return None
    text = str(value)
    if text == "":
        return None
    return hashlib.sha256(f"{salt}{text}".encode("utf-8")).hexdigest()


def coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def request_json_with_retry(
    session: requests.Session,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    timeout_sec: float = 20,
    retry_config: Mapping[str, Any] | None = None,
    logger: logging.Logger | None = None,
    context: str = "request",
) -> dict[str, Any]:
    retry_config = retry_config or {}
    retry_statuses = set(retry_config.get("status_forcelist", [429, 500, 502, 503, 504]))
    max_attempts = int(retry_config.get("max_attempts", 5))
    backoff = float(retry_config.get("backoff_initial_sec", 1.0))
    multiplier = float(retry_config.get("backoff_multiplier", 2.0))
    max_backoff = float(retry_config.get("backoff_max_sec", 30.0))
    jitter = float(retry_config.get("jitter_sec", 0.25))
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(url, params=params, timeout=timeout_sec)
            status_code = response.status_code

            if status_code == 403:
                raise SteamForbiddenError(f"{context}: HTTP 403 forbidden")

            if status_code in retry_statuses:
                last_error = SteamRequestError(f"{context}: HTTP {status_code}")
                if attempt == max_attempts:
                    break
                sleep_for = min(backoff, max_backoff) + random.uniform(0, jitter)
                if logger:
                    logger.warning(
                        "%s failed with HTTP %s on attempt %s/%s; sleeping %.2fs",
                        context,
                        status_code,
                        attempt,
                        max_attempts,
                        sleep_for,
                    )
                time.sleep(sleep_for)
                backoff *= multiplier
                continue

            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise SteamRequestError(f"{context}: JSON payload is not an object")
            return payload

        except SteamForbiddenError:
            raise
        except requests.RequestException as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            sleep_for = min(backoff, max_backoff) + random.uniform(0, jitter)
            if logger:
                logger.warning(
                    "%s request error on attempt %s/%s: %s; sleeping %.2fs",
                    context,
                    attempt,
                    max_attempts,
                    exc,
                    sleep_for,
                )
            time.sleep(sleep_for)
            backoff *= multiplier
        except ValueError as exc:
            raise SteamRequestError(f"{context}: invalid JSON response") from exc

    raise SteamRetryExceeded(f"{context} failed after {max_attempts} attempts: {last_error}")
