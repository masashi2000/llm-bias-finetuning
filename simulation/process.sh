#!/bin/bash

# スクリプトの開始を表示
echo "Starting batch scripts..."

# Pythonスクリプトのパスを指定
# 1つ目のコマンドの実行
echo "RACISM 2, 0"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 2 --num_republican_agents 0 --instruction_file ./files/echo_racism.txt

echo "RACISM 0, 2"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 0 --num_republican_agents 2 --instruction_file ./files/echo_racism.txt
echo "RACISM 1, 1"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 1 --num_republican_agents 1 --instruction_file ./files/echo_racism.txt

# 1つ目のコマンドの実行
echo "ILLEGAL IMMIGRATION 2, 0"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 2 --num_republican_agents 0 --instruction_file ./files/echo_illegal_immigration.txt

echo "ILLEGAL IMMIGRATION 0, 2"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 0 --num_republican_agents 2 --instruction_file ./files/echo_illegal_immigration.txt
echo "ILLEGAL IMMIGRATION 1, 1"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 1 --num_republican_agents 1 --instruction_file ./files/echo_illegal_immigration.txt

# 2つ目のコマンドの実行
echo "GUN VIOLENCE 2, 0"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 2 --num_republican_agents 0 --instruction_file ./files/echo_gun_violence.txt

echo "GUN VIOLENCE 0, 2"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 0 --num_republican_agents 2 --instruction_file ./files/echo_gun_violence.txt

echo "GUN VIOLENCE 1, 1"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 1 --num_republican_agents 1 --instruction_file ./files/echo_gun_violence.txt

# 2つ目のコマンドの実行
echo "CLIMATE CHANGE 2, 0"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 2 --num_republican_agents 0 --instruction_file ./files/echo_climate_change.txt

echo "CLIMATE CHANGE 0, 2"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 0 --num_republican_agents 2 --instruction_file ./files/echo_climate_change.txt

echo "CLIMATE CHANGE 1, 1"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 1 --num_republican_agents 1 --instruction_file ./files/echo_climate_change.txt

# 終了メッセージを表示
echo "Batch scripts completed."
