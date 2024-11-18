import pandas as pd
import argparse
import os

def main():
    # コマンドライン引数を設定
    parser = argparse.ArgumentParser(description="Two CSV files are merged into one.")
    parser.add_argument("file1", help="Path to the first CSV file")
    parser.add_argument("file2", help="Path to the second CSV file")
    args = parser.parse_args()
    
    # CSVファイルを読み込む
    csv1 = pd.read_csv(args.file1)
    csv2 = pd.read_csv(args.file2)
    
    # データを結合（行方向に結合: vertical stack）
    combined = pd.concat([csv1, csv2], ignore_index=True)
    
    # 出力ファイル名を作成
    base_name1 = os.path.splitext(os.path.basename(args.file1))[0]
    base_name2 = os.path.splitext(os.path.basename(args.file2))[0]
    output_file_name = f"{base_name1}_{base_name2}.csv"
    
    # 結果をCSVとして保存
    combined.to_csv(output_file_name, index=False)
    print(f"Files merged and saved as {output_file_name}")

if __name__ == "__main__":
    main()

