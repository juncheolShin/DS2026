import csv
import json
from itertools import zip_longest

# =========================
# 파일 이름 설정
# =========================

game_csv_file = "game_names.csv"        # 전에 만든 게임 이름 CSV
annotation_jsonl_file = "reviews.jsonl" # 리뷰 + annotation JSONL
output_csv_file = "game_names_with_scores.csv"


# =========================
# 태그 목록
# =========================

TAGS = [
    "최적화#프레임",
    "최적화#조작감",
    "시스템#버그",
    "콘텐츠#볼륨",
    "콘텐츠#몰입도",
    "시스템#밸런스",
    "시스템#독창성",
    "시스템#자유도",
    "UX#그래픽",
    "UX#캐릭터디자인",
    "UX#사운드",
    "스토리#내러티브",
    "운영#핵/치트",
    "운영#업데이트",
    "콘텐츠#난이도",
    "시스템#진입장벽",
    "콘텐츠#피로도",
]

SENTIMENT_TO_SCORE = {
    "positive": 1,
    "neutral": 0,
    "negative": -1,
}


# =========================
# 기존 게임 CSV 읽기
# =========================

with open(game_csv_file, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    game_rows = list(reader)
    original_columns = reader.fieldnames

if original_columns is None:
    raise ValueError("게임 CSV에 헤더가 없습니다.")


# =========================
# annotation JSONL 읽기
# =========================

with open(annotation_jsonl_file, "r", encoding="utf-8") as f:
    annotation_lines = f.readlines()


# =========================
# 줄 수 검사
# =========================

if len(game_rows) != len(annotation_lines):
    print("경고: 게임 CSV 줄 수와 JSONL 줄 수가 다릅니다.")
    print(f"게임 CSV 데이터 수: {len(game_rows)}")
    print(f"JSONL 데이터 수: {len(annotation_lines)}")
    print("줄 번호 기준으로 가능한 만큼만 처리합니다.")


# =========================
# 병합 처리
# =========================

result_rows = []

json_errors = []
unknown_tags = {}
unknown_sentiments = {}
conflicts = []

for line_num, (game_row, json_line) in enumerate(
    zip_longest(game_rows, annotation_lines),
    start=1
):
    if game_row is None or json_line is None:
        break

    new_row = dict(game_row)

    # 기본값은 빈칸
    # 빈칸 = 해당 리뷰에서 그 속성 언급 없음
    for tag in TAGS:
        new_row[tag] = ""

    json_line = json_line.strip()

    if not json_line:
        result_rows.append(new_row)
        continue

    try:
        data = json.loads(json_line)
    except json.JSONDecodeError:
        json_errors.append(line_num)
        result_rows.append(new_row)
        continue

    annotations = data.get("annotation", [])

    for item in annotations:
        if not isinstance(item, list) or len(item) != 2:
            continue

        tag, sentiment = item

        if tag not in TAGS:
            unknown_tags[tag] = unknown_tags.get(tag, 0) + 1
            continue

        if sentiment not in SENTIMENT_TO_SCORE:
            unknown_sentiments[sentiment] = unknown_sentiments.get(sentiment, 0) + 1
            continue

        score = SENTIMENT_TO_SCORE[sentiment]

        # 같은 리뷰 안에서 같은 태그가 여러 번 나온 경우 처리
        if new_row[tag] == "":
            new_row[tag] = score
        else:
            previous_score = new_row[tag]

            # 같은 값이면 그냥 무시
            if previous_score == score:
                continue

            # 서로 다른 감정이 같은 태그에 들어오면 충돌 기록
            conflicts.append({
                "line": line_num,
                "tag": tag,
                "previous_score": previous_score,
                "new_score": score,
                "sentence_form": data.get("sentence_form", "")
            })

            # 일단 기존 값을 유지
            # 바꾸고 싶으면 아래 줄을 활성화하면 됨
            # new_row[tag] = score

    result_rows.append(new_row)


# =========================
# CSV 저장
# =========================

output_columns = original_columns + TAGS

with open(output_csv_file, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=output_columns)
    writer.writeheader()
    writer.writerows(result_rows)


# =========================
# 결과 출력
# =========================

print(f"완료! {output_csv_file} 파일이 생성되었습니다.")
print(f"처리된 리뷰 수: {len(result_rows)}")

if json_errors:
    print(f"JSON 파싱 오류 줄 수: {len(json_errors)}")
    print("오류 줄 번호 예시:", json_errors[:10])

if unknown_tags:
    print("목록에 없는 태그 발견:")
    for tag, count in unknown_tags.items():
        print(f"  {tag}: {count}개")

if unknown_sentiments:
    print("목록에 없는 감정값 발견:")
    for sentiment, count in unknown_sentiments.items():
        print(f"  {sentiment}: {count}개")

if conflicts:
    print(f"같은 리뷰 안에서 감정 충돌 발생: {len(conflicts)}개")
    print("충돌 예시:")
    for c in conflicts[:5]:
        print(c)