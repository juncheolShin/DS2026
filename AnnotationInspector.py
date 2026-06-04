import json
from collections import Counter
from pathlib import Path


DATA_FILE = "steam_reviews_pre_labeled_GPT.jsonl"

ENTITY_PROPERTY_PAIR = [
    '최적화#프레임',
    '최적화#조작감',
    '시스템#버그',
    '콘텐츠#볼륨',
    '콘텐츠#몰입도',
    '콘텐츠#난이도',
    '콘텐츠#피로도',
    '시스템#밸런스',
    '시스템#독창성',
    '시스템#자유도',
    '시스템#진입장벽',
    'UX#그래픽',
    'UX#캐릭터디자인',
    'UX#사운드',
    '스토리#내러티브',
    '운영#핵/치트',
    '운영#업데이트'
]

VALID_POLARITIES = {
    "positive",
    "negative",
    "neutral",
    "------------"
}


def short_text(text, limit=100):
    text = str(text).replace("\n", " ").replace("\r", " ")
    return text[:limit] + "..." if len(text) > limit else text


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append((line_no, json.loads(line)))
            except json.JSONDecodeError as e:
                rows.append((line_no, {
                    "__json_error__": str(e),
                    "__raw_line__": line
                }))

    return rows


def check_annotations(rows):
    valid_tags = set(ENTITY_PROPERTY_PAIR)

    errors = []
    warnings = []

    tag_counter = Counter()
    polarity_counter = Counter()
    unknown_tag_counter = Counter()
    unknown_polarity_counter = Counter()
    annotation_length_counter = Counter()

    total_rows = 0
    total_annotations = 0
    empty_annotation_rows = 0

    for line_no, row in rows:
        total_rows += 1

        if "__json_error__" in row:
            errors.append({
                "line": line_no,
                "type": "JSON_PARSE_ERROR",
                "message": row["__json_error__"],
                "raw": short_text(row.get("__raw_line__", ""))
            })
            continue

        if not isinstance(row, dict):
            errors.append({
                "line": line_no,
                "type": "ROW_NOT_OBJECT",
                "message": "각 줄은 JSON object여야 합니다.",
                "row": short_text(row)
            })
            continue

        sentence = row.get("sentence_form", "")

        if "annotation" not in row:
            errors.append({
                "line": line_no,
                "type": "MISSING_ANNOTATION",
                "message": "annotation 필드가 없습니다.",
                "sentence": short_text(sentence)
            })
            continue

        annotations = row["annotation"]

        if not isinstance(annotations, list):
            errors.append({
                "line": line_no,
                "type": "ANNOTATION_NOT_LIST",
                "message": "annotation은 list여야 합니다.",
                "annotation": short_text(annotations),
                "sentence": short_text(sentence)
            })
            continue

        if len(annotations) == 0:
            empty_annotation_rows += 1
            continue

        seen_tags = set()

        for ann_idx, ann in enumerate(annotations, start=1):
            total_annotations += 1

            if not isinstance(ann, list):
                errors.append({
                    "line": line_no,
                    "annotation_index": ann_idx,
                    "type": "ANNOTATION_ITEM_NOT_LIST",
                    "message": "각 annotation 항목은 list여야 합니다.",
                    "annotation": short_text(ann),
                    "sentence": short_text(sentence)
                })
                continue

            annotation_length_counter[len(ann)] += 1

            if len(ann) < 2:
                errors.append({
                    "line": line_no,
                    "annotation_index": ann_idx,
                    "type": "ANNOTATION_TOO_SHORT",
                    "message": "annotation은 최소 [속성명, 감성] 형태여야 합니다.",
                    "annotation": short_text(ann),
                    "sentence": short_text(sentence)
                })
                continue

            if len(ann) > 2:
                warnings.append({
                    "line": line_no,
                    "annotation_index": ann_idx,
                    "type": "ANNOTATION_HAS_EXTRA_FIELDS",
                    "message": "annotation 길이가 2보다 큽니다. 현재 학습 코드는 annotation[0], annotation[1]만 사용합니다.",
                    "annotation": short_text(ann),
                    "sentence": short_text(sentence)
                })

            tag = ann[0]
            polarity = ann[1]

            if tag not in valid_tags:
                unknown_tag_counter[str(tag)] += 1
                errors.append({
                    "line": line_no,
                    "annotation_index": ann_idx,
                    "type": "UNKNOWN_TAG",
                    "message": f"정의되지 않은 속성명입니다: {tag}",
                    "annotation": short_text(ann),
                    "sentence": short_text(sentence)
                })
            else:
                tag_counter[tag] += 1

            if polarity not in VALID_POLARITIES:
                unknown_polarity_counter[str(polarity)] += 1
                errors.append({
                    "line": line_no,
                    "annotation_index": ann_idx,
                    "type": "UNKNOWN_POLARITY",
                    "message": f"정의되지 않은 감성값입니다: {polarity}",
                    "annotation": short_text(ann),
                    "sentence": short_text(sentence)
                })
            else:
                polarity_counter[polarity] += 1

            if tag in seen_tags:
                warnings.append({
                    "line": line_no,
                    "annotation_index": ann_idx,
                    "type": "DUPLICATE_TAG_IN_SAME_REVIEW",
                    "message": f"한 리뷰 안에서 같은 속성이 중복되었습니다: {tag}",
                    "annotation": short_text(ann),
                    "sentence": short_text(sentence)
                })

            seen_tags.add(tag)

    print("\n========== Annotation 검사 결과 ==========")
    print(f"파일명: {DATA_FILE}")
    print(f"전체 리뷰 row 수: {total_rows}")
    print(f"전체 annotation 수: {total_annotations}")
    print(f"annotation이 빈 리뷰 수: {empty_annotation_rows}")
    print(f"에러 수: {len(errors)}")
    print(f"경고 수: {len(warnings)}")

    print("\n========== annotation 길이 분포 ==========")
    for length, count in sorted(annotation_length_counter.items()):
        print(f"길이 {length}: {count}")

    print("\n========== 감성값 분포 ==========")
    for polarity, count in polarity_counter.most_common():
        print(f"{polarity}: {count}")

    print("\n========== 태그별 annotation 개수 ==========")
    for tag in ENTITY_PROPERTY_PAIR:
        print(f"{tag}: {tag_counter[tag]}")

    if unknown_tag_counter:
        print("\n========== 알 수 없는 속성명 ==========")
        for tag, count in unknown_tag_counter.most_common():
            print(f"{tag}: {count}")

    if unknown_polarity_counter:
        print("\n========== 알 수 없는 감성값 ==========")
        for polarity, count in unknown_polarity_counter.most_common():
            print(f"{polarity}: {count}")

    if warnings:
        print("\n========== 경고 샘플 ==========")
        for warning in warnings[:30]:
            print(json.dumps(warning, ensure_ascii=False, indent=2))

    if errors:
        print("\n========== 에러 샘플 ==========")
        for error in errors[:50]:
            print(json.dumps(error, ensure_ascii=False, indent=2))

        print("\n❌ 검사 실패")
        print("위 에러를 먼저 수정해야 학습 중 라벨 문제가 줄어듭니다.")
    else:
        print("\n✅ 검사 통과")
        print("annotation의 속성명과 감성값 형식은 정상으로 보입니다.")


def main():
    path = Path(DATA_FILE)

    if not path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {path.resolve()}")
        print("check_annotations.py와 steam_reviews_pre_labeled_GPT.jsonl이 같은 폴더에 있는지 확인하세요.")
        return

    rows = load_jsonl(path)
    check_annotations(rows)


if __name__ == "__main__":
    main()