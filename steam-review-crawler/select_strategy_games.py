import os
import re
import shutil
import time
import urllib.request
import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def backup_existing_files():
    print("=== Backing up existing files ===")
    project_root = Path(__file__).parent.resolve()
    data_dir = project_root / "data"
    backup_dir = project_root / "Label-data"

    if not data_dir.exists():
        print("No data directory found. Nothing to backup.")
        return

    # Create backup directories
    backup_dir.mkdir(exist_ok=True)
    (backup_dir / "input").mkdir(exist_ok=True)
    (backup_dir / "raw").mkdir(exist_ok=True)

    # Backup appids.csv
    appids_src = data_dir / "input" / "appids.csv"
    if appids_src.exists():
        shutil.copy2(appids_src, backup_dir / "input" / "appids.csv")
        print(f"Backed up: {appids_src} -> {backup_dir / 'input' / 'appids.csv'}")

    # Backup raw directory contents
    raw_dir_src = data_dir / "raw"
    if raw_dir_src.exists():
        for item in raw_dir_src.iterdir():
            if item.is_file():
                shutil.copy2(item, backup_dir / "raw" / item.name)
                print(f"Backed up: {item} -> {backup_dir / 'raw' / item.name}")
    print("Backup complete!\n")

def fetch_strategy_search_page(start, count=100):
    url = f"https://store.steampowered.com/search/results/?tags=9&start={start}&count={count}&cc=kr"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
    )
    with urllib.request.urlopen(req) as response:
        return response.read().decode('utf-8')

def get_korean_reviews_info(appid):
    url = f"https://store.steampowered.com/appreviews/{appid}?json=1&language=koreana&purchase_type=all&review_type=all"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
    )
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            summary = data.get("query_summary", {})
            return {
                "total_reviews": summary.get("total_reviews", 0),
                "total_positive": summary.get("total_positive", 0)
            }
    except Exception as e:
        print(f"Error fetching reviews for appid {appid}: {e}")
        return None

def main():
    backup_existing_files()

    print("=== Fetching Strategy candidates from Steam Store search ===")
    candidates = []
    # Fetch 600 games (start from 0 to 500)
    for start in range(0, 600, 100):
        print(f"Fetching search results start={start}...")
        try:
            html = fetch_strategy_search_page(start, 100)
            pattern = r'<a\s+href="https://store\.steampowered\.com/app/\d+/[^"]*".*?class="[^"]*search_result_row.*?</a>'
            matches = list(re.finditer(pattern, html, re.DOTALL))
            for match in matches:
                block = match.group(0)
                appid_match = re.search(r'href="https://store\.steampowered\.com/app/(\d+)/', block)
                appid = int(appid_match.group(1)) if appid_match else None
                
                title_match = re.search(r'<span class="title">(.*?)</span>', block)
                title = title_match.group(1).strip() if title_match else "unknown"
                
                tooltip_match = re.search(r'data-tooltip-html="([^"]+)"', block)
                tooltip_text = tooltip_match.group(1) if tooltip_match else ""
                tooltip_text = tooltip_text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&amp;", "&")
                
                count_match = re.search(r'of\s+the\s+([\d,]+)\s+user\s+reviews', tooltip_text)
                if not count_match:
                    count_match = re.search(r'of\s+the\s+([\d,]+)\s+reviews', tooltip_text)
                
                global_reviews = int(count_match.group(1).replace(',', '')) if count_match else 0
                
                if appid:
                    candidates.append({
                        "appid": appid,
                        "name": title,
                        "global_reviews": global_reviews
                    })
            time.sleep(0.5)
        except Exception as e:
            print(f"Failed to fetch search page start={start}: {e}")

    print(f"Total candidates fetched from search: {len(candidates)}")

    # Pre-filter: only keep games with at least 300 global reviews to avoid completely empty ones
    pre_filtered = [c for c in candidates if c["global_reviews"] >= 300]
    print(f"Candidates with >= 300 global reviews: {len(pre_filtered)}")

    # Query Steam Reviews API to get actual Korean review count and positive review percentage
    valid_candidates = []
    print("\n=== Checking Korean review count for candidates ===")
    for i, game in enumerate(pre_filtered):
        appid = game["appid"]
        name = game["name"]
        print(f"[{i+1}/{len(pre_filtered)}] Checking AppID {appid} ({name})... ", end="", flush=True)
        
        info = get_korean_reviews_info(appid)
        if info:
            total_korean = info["total_reviews"]
            total_pos = info["total_positive"]
            if total_korean >= 100:
                pos_percent = (total_pos / total_korean) * 100 if total_korean > 0 else 0
                valid_candidates.append({
                    "appid": appid,
                    "name": name,
                    "korean_reviews": total_korean,
                    "korean_positive_percent": pos_percent
                })
                print(f"Valid! Korean reviews: {total_korean}, Positive: {pos_percent:.2f}%")
            else:
                print(f"Skipped. Only {total_korean} Korean reviews.")
        else:
            print("Failed to fetch.")
        
        # Sleep to avoid rate limiting
        time.sleep(0.15)

    print(f"\nTotal valid candidates with >= 100 Korean reviews: {len(valid_candidates)}")

    if len(valid_candidates) < 100:
        print(f"Warning: Only found {len(valid_candidates)} valid candidates. We will use all of them.")
        sampled = valid_candidates
    else:
        # Sort by Korean positive percentage
        valid_candidates.sort(key=lambda x: x["korean_positive_percent"])
        
        # Systematic sampling to get exactly 100 games
        sampled = []
        for j in range(100):
            idx = int(round(j * (len(valid_candidates) - 1) / 99.0))
            sampled.append(valid_candidates[idx])

    # Save to data/input/appids.csv
    project_root = Path(__file__).parent.resolve()
    output_path = project_root / "data" / "input" / "appids.csv"
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write("appid,name,korean_reviews,korean_positive_percent\n")
        for game in sampled:
            # Escape quotes in game name
            clean_name = game["name"].replace('"', '""')
            f.write(f"{game['appid']},\"{clean_name}\",{game['korean_reviews']},{game['korean_positive_percent']:.2f}\n")

    print(f"\nSuccessfully saved {len(sampled)} games to {output_path}")
    print("\nSample of selected games (lowest, middle, highest ratings):")
    if sampled:
        print(f"Worst Rating: AppID {sampled[0]['appid']} | {sampled[0]['name']} | {sampled[0]['korean_positive_percent']:.2f}% ({sampled[0]['korean_reviews']} reviews)")
        mid = len(sampled) // 2
        print(f"Median Rating: AppID {sampled[mid]['appid']} | {sampled[mid]['name']} | {sampled[mid]['korean_positive_percent']:.2f}% ({sampled[mid]['korean_reviews']} reviews)")
        print(f"Best Rating: AppID {sampled[-1]['appid']} | {sampled[-1]['name']} | {sampled[-1]['korean_positive_percent']:.2f}% ({sampled[-1]['korean_reviews']} reviews)")

if __name__ == "__main__":
    main()
