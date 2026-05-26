# Steam Review Crawler

Steam 리뷰를 게임 단위 분석용 원천 데이터셋으로 수집하는 Python 3.11 프로젝트입니다. 최종 분석은 ABSA(Aspect-Based Sentiment Analysis)를 전제로 하며, 리뷰 텍스트와 리뷰 메타데이터, 게임 메타데이터, 게임별 리뷰 요약을 분리 저장합니다.

## 설치 방법

```bash
python -m venv .venv
source .venv/bin/activate  # Windows는 .venv\Scripts\activate
pip install -r requirements.txt
```

선택 사항으로 `.env.example`을 참고해 `.env`를 만들 수 있습니다. `STEAM_REVIEW_HASH_SALT`는 작성자 steamid를 sha256 해시로 바꾸기 전에 붙이는 salt입니다.

## config.yaml 설명

- `paths.appids_csv`: 수집 대상 appid CSV 경로입니다.
- `crawler.language`: `koreana` 또는 `all`을 사용할 수 있습니다. 기본값은 한국어 리뷰용 `koreana`입니다.
- `crawler.filter`: 기본값은 `recent`입니다. `all`은 helpfulness 기반 sliding window 때문에 전체 순회 종료 조건이 애매할 수 있어 기본값으로 두지 않았습니다.
- `crawler.num_per_page`: Steam 리뷰 API 페이지당 요청 수입니다. 최대 100으로 제한됩니다.
- `crawler.max_reviews_per_app`: 게임당 최대 수집 리뷰 수입니다. 기본 설정은 학습용 수집 기준 1,000개입니다.
- `crawler.sleep_sec`: 페이지 요청 사이 대기 시간입니다.
- `crawler.max_failures`: 한 appid에서 연속 실패가 이 값을 초과하면 수집을 종료합니다.
- `crawler.retry.status_forcelist`: `429, 500, 502, 503, 504`가 retry 대상입니다. `403`은 즉시 중단됩니다.
- `metadata.enabled`: Steam appdetails 메타데이터 수집 여부입니다. 실패해도 리뷰 수집은 계속 진행됩니다.
- `output.save_csv`, `output.save_parquet`: CSV와 Parquet 저장 여부입니다.

## appids.csv 예시

`data/input/appids.csv`:

```csv
appid,name,target_genre,priority
413150,Stardew Valley,simulation,1
1245620,ELDEN RING,action RPG,2
367520,Hollow Knight,metroidvania,3
578080,PUBG: BATTLEGROUNDS,battle royale,4
730,Counter-Strike 2,FPS,5
```

처음부터 전체 Steam을 순회하지 않고 이 파일에 지정된 게임만 수집합니다. 확장을 위해 `src/app_list.py`에 전체 app list 수집 함수도 분리해 두었습니다.

## 실행 명령어

```bash
python main.py --config config.yaml
python main.py --config config.yaml --test-mode
python main.py --config config.yaml --language all --max-reviews-per-app 500
python src/validate_dataset.py --reviews data/raw/reviews_raw.parquet
```

수집 결과는 기본적으로 다음 위치에 저장됩니다.

- `data/raw/reviews_raw.parquet`, `data/raw/reviews_raw.csv`
- `data/raw/review_summary_by_app.parquet`, `data/raw/review_summary_by_app.csv`
- `data/raw/app_metadata.parquet`, `data/raw/app_metadata.csv`
- `data/logs/failed_appids.csv`
- `data/logs/crawler.log`

## 수집되는 컬럼

`reviews_raw`는 리뷰 1개가 1행입니다.

- `appid`, `game_name`, `recommendationid`, `language`, `review`
- `is_empty_review`: null 또는 빈 리뷰 여부입니다. 빈 리뷰도 삭제하지 않습니다.
- `voted_up`, `timestamp_created`, `timestamp_updated`
- `steam_purchase`, `received_for_free`, `written_during_early_access`
- `votes_up`, `votes_funny`, `weighted_vote_score`, `comment_count`
- `author_steamid_hash`: steamid 원문은 저장하지 않고 sha256 해시만 저장합니다.
- `author_num_games_owned`, `author_num_reviews`
- `playtime_forever`, `playtime_last_two_weeks`, `playtime_at_review`, `last_played`
- `crawled_at`

`review_summary_by_app`는 게임 1개가 1행입니다.

- `appid`, `game_name`
- `total_reviews_api`, `total_positive_api`, `total_negative_api`
- `review_score`, `review_score_desc`
- `collected_reviews_count`, `positive_collected_count`, `negative_collected_count`
- `first_review_time`, `last_review_time`, `crawled_at`
- `termination_reason`: `no_reviews`, `max_reviews_per_app`, `cursor_loop`, `max_failures`, `http_403` 등

`app_metadata`는 게임 1개가 1행입니다.

- `appid`, `name`, `release_date`
- `developers`, `publishers`, `genres`, `categories`
- `price_overview`, `metacritic_score`, `is_free`, `required_age`
- `short_description`, `header_image`, `metadata_source`, `crawled_at`

리스트와 딕셔너리 성격의 메타데이터는 CSV와 Parquet 모두에서 안정적으로 다루기 위해 JSON 문자열로 저장합니다.

## Steam API 제약사항

- 리뷰 endpoint는 `GET https://store.steampowered.com/appreviews/{appid}?json=1`을 사용합니다.
- 첫 요청은 `cursor="*"`로 시작하고, 응답의 `cursor`를 다음 요청에 넘깁니다.
- cursor는 URL encoding이 필요할 수 있으므로 `requests.get(..., params=params)` 방식으로 전달합니다.
- 기본 파라미터는 `filter=recent`, `language=koreana`, `review_type=all`, `purchase_type=all`, `num_per_page=100`, `filter_offtopic_activity=1`입니다.
- 한 appid 수집은 리뷰 배열이 비거나, 최대 수집 수에 도달하거나, 같은 cursor가 반복되거나, 연속 실패가 `max_failures`를 초과하면 종료됩니다.
- Steam 공식 리뷰 API는 판매량을 직접 제공하지 않습니다. 따라서 판매량 분석에는 `total_reviews_api`, `total_positive_api`, `total_negative_api` 같은 리뷰 수 지표를 proxy로 사용하는 한계가 있습니다.
- `filter=recent` 응답에서 진짜 전체 리뷰 수인 `total_reviews`가 제공되지 않는 경우 `total_reviews_api`는 비워 둡니다. 이때 실제 확보한 리뷰 수는 `collected_reviews_count`를 기준으로 확인합니다.
- `appdetails` endpoint는 Steamworks 공식 Web API 문서화 범위가 제한적이므로 실패할 수 있습니다. 이 경우 `metadata_source="failed"`로 저장하고 리뷰 수집은 계속합니다.

## ABSA 분석에서 활용할 수 있는 컬럼

- `review`: 자유도, 액션감, 최적화, 스토리, 조작감, 퍼즐 요소 등 aspect phrase 추출 원문입니다.
- `voted_up`: 리뷰 단위 전체 긍정/부정 라벨입니다.
- `language`: 한국어 전용 분석과 다국어 확장을 분리할 수 있습니다.
- `playtime_at_review`, `playtime_forever`: 너무 낮은 플레이타임 리뷰를 후처리에서 필터링할 때 사용합니다.
- `timestamp_created`, `timestamp_updated`: 최신 리뷰와 과거 리뷰의 시점 차이를 통제할 때 사용합니다.
- `release_date`: 너무 오래된 게임을 제외하거나 출시 후 경과 기간을 통제할 때 사용합니다.
- `genres`, `categories`, `price_overview`, `metacritic_score`: 게임 단위 통제 변수로 사용할 수 있습니다.
- `total_reviews_api`, `review_score`, `review_score_desc`: Steam 리뷰 API가 제공하는 게임 단위 평판 지표입니다.

## 검증

```bash
python src/validate_dataset.py --reviews data/raw/reviews_raw.parquet
```

검증 스크립트는 recommendationid 중복 수, appid별 리뷰 수, `voted_up` 분포, `playtime_at_review` 결측률, 리뷰 텍스트 결측률, `timestamp_created` 범위, API 총 리뷰 수와 수집 수 비교, 실패 appid 목록, 샘플 리뷰 20개를 콘솔에 출력합니다.
