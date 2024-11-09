# Examples: ./run_analysis.sh /mnt/c/Users/sakurai/llm-bias-finetuning/simulation/data_normal_2
#!/bin/bash

# 引数で処理対象ディレクトリのパスを指定
data_dir="$1"
analyze_dir="/mnt/c/Users/sakurai/llm-bias-finetuning/analyze"

# ディレクトリパスが指定されていない場合はエラーメッセージを表示して終了
if [ -z "$data_dir" ]; then
    echo "Error: 処理対象のディレクトリパスを指定してください。"
    echo "Usage: $0 /path/to/target_directory"
    exit 1
fi

# 入力ディレクトリの最後のディレクトリ名を取得して出力ディレクトリを作成
output_dir="$analyze_dir/$(basename "$data_dir")"
mkdir -p "$output_dir"  # 出力ディレクトリを作成

# analyzeディレクトリに移動
cd "$analyze_dir"

# data_dir内の各ディレクトリをループ
for dir in "$data_dir"/*; do
    # ディレクトリかどうか確認
    if [ -d "$dir" ]; then
        # ターゲットファイルとディレクトリ名を取得
        target_file="$dir/survey_results.csv"
        dir_name=$(basename "$dir")

        # ターゲットファイルが存在するか確認してからPythonスクリプトを実行
        if [ -f "$target_file" ]; then
            python analyze2.py --target_file "$target_file" --save_name "$output_dir/$dir_name"
        else
            echo "File not found: $target_file"
        fi
    fi
done

