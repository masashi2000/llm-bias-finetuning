# python combine_images.py /path/to/image_folder /path/to/output_directory
import os
import glob
import argparse
from PIL import Image

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description="画像をグリッド形式で結合します。")
parser.add_argument("image_folder", type=str, help="画像が保存されているフォルダのパス。")
parser.add_argument("output_directory", type=str, help="結合画像を保存するディレクトリ。")

# 引数の解析
args = parser.parse_args()
image_folder = args.image_folder
output_directory = args.output_directory
output_image_path = os.path.join(output_directory, "combined_image.png")

# グリッドの設定
rows = 4  # 行数（トピック数）
cols = 3  # 列数（パターン数）

# トピックとパターンの順序
topics = ["illegal_immigration", "racism", "gun_violence", "climate_change"]
party_patterns = ["Dem2_Rep0", "Dem1_Rep1", "Dem0_Rep2"]

# 画像の読み込みと最小サイズの取得
images = []
min_width, min_height = float('inf'), float('inf')

for topic in topics:
    row_images = []
    for pattern in party_patterns:
        search_pattern = f"*_{topic}_{pattern}_Round*_Trial*.png"
        file_path = glob.glob(os.path.join(image_folder, search_pattern))

        if not file_path:
            print(f"トピック '{topic}' とパターン '{pattern}' に一致するファイルが見つかりません。")
            row_images.append(None)
            continue

        image = Image.open(file_path[0])
        row_images.append(image)

        # 最小の幅と高さを更新
        if image.width < min_width:
            min_width = image.width
        if image.height < min_height:
            min_height = image.height

    images.append(row_images)

# 画像のリサイズ
resized_images = []
for row in images:
    resized_row = [img.resize((min_width, min_height), Image.LANCZOS) if img else None for img in row]
    resized_images.append(resized_row)

# 結合画像のキャンバス作成
combined_width = min_width * cols
combined_height = min_height * rows
combined_image = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))  # 白背景

# 画像の貼り付け
for i, row in enumerate(resized_images):
    for j, image in enumerate(row):
        if image:
            x = j * min_width
            y = i * min_height
            combined_image.paste(image, (x, y))

# 結合画像の保存
combined_image.save(output_image_path)
print(f"結合画像が {output_image_path} に保存されました。")

