import csv
import json
import sys
import os

def convert_csv_to_json(input_file):
    # 入力ファイルの拡張子を除いた名前を取得し、.jsonを追加
    output_file = os.path.splitext(input_file)[0] + ".json"

    try:
        with open(input_file, mode='r', encoding='utf-8') as csv_file, open(output_file, mode='w', encoding='utf-8') as json_file:
            reader = csv.DictReader(csv_file)

            # CSV内に"Answer"と"Response"のカラムがあるかチェック
            if 'Answer' not in reader.fieldnames or 'Response' not in reader.fieldnames:
                raise ValueError("CSVファイルには'Answer'または'Response'のカラムが含まれていません。")

            # 各行をフォーマットしてJSONに書き込む
            for row in reader:
                record = {
                    "prompt": row['Answer'],
                    "completion": row['Response']
                }
                json_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"変換が成功しました。出力ファイル: {output_file}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    # コマンドライン引数の確認
    if len(sys.argv) != 2:
        print("使用法: python script.py <入力CSVファイル>")
        sys.exit(1)

    input_csv_file = sys.argv[1]
    convert_csv_to_json(input_csv_file)

