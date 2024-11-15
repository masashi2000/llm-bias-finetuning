#!/bin/bash

# 実行するPythonスクリプトを変数に定義
script1="python generate_persona.py prompt_file_democrat_v2.txt"
script2="python generate_persona.py prompt_file_republican_v2.txt"

# スクリプト1を実行
echo "Running: $script1"
$script1

# スクリプト2を実行
echo "Running: $script2"
$script2

# 完了メッセージ
echo "Both scripts have finished executing."

