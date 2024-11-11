#!/bin/bash

# ファイルのリスト
files=("test_llama_3_2_1b.py" "test_llama_3_2_3b.py" "test_mistral_7b_v3.py" "test_mistral_8b.py")

# 各ファイルを実行
for file in "${files[@]}"; do
    echo "Executing $file..."
    python3 "$file"
    if [ $? -ne 0 ]; then
        echo "Error executing $file. Stopping."
        exit 1
    fi
done

echo "All scripts executed successfully."

