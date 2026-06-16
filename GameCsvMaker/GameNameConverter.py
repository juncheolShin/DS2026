import csv

input_file = "reviews.csv"
output_file = "game_names.csv"

with open(input_file, "r", encoding="utf-8-sig", newline="") as f_in, \
     open(output_file, "w", encoding="utf-8-sig", newline="") as f_out:

    reader = csv.DictReader(f_in)
    writer = csv.writer(f_out)

    writer.writerow(["game_name"])

    for row_num, row in enumerate(reader, start=2):
        game_name = row.get("game_name", "").strip()

        if game_name:
            writer.writerow([game_name])
        else:
            print(f"{row_num}번째 줄: game_name 없음, 건너뜀")

print(f"완료! {output_file} 파일이 생성되었습니다.")