#!/bin/bash
# nohup ./process.sh > output2.log 2>&1 &

# スクリプトの開始を表示
echo "Starting batch scripts..."

echo "NORMAL RACISM 2, 0"
python main_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 2 --num_republican_agents 0 --instruction_file ./files/racism.txt

echo "NORMAL RACISM 0, 2"
python main_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 0 --num_republican_agents 2 --instruction_file ./files/racism.txt
echo "NORMAL RRACISM 1, 1"
python main_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 1 --num_republican_agents 1 --instruction_file ./files/racism.txt

# 1つ目のコマンドの実行
echo "NORMAL ILLEGAL IMMIGRATION 2, 0"
python main_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 2 --num_republican_agents 0 --instruction_file ./files/illegal_immigration.txt

echo "NORMAL ILLEGAL IMMIGRATION 0, 2"
python main_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 0 --num_republican_agents 2 --instruction_file ./files/illegal_immigration.txt
echo "NORMAL ILLEGAL IMMIGRATION 1, 1"
python main_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 1 --num_republican_agents 1 --instruction_file ./files/illegal_immigration.txt

# 2つ目のコマンドの実行
echo "NORMAL GUN VIOLENCE 2, 0"
python main_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 2 --num_republican_agents 0 --instruction_file ./files/gun_violence.txt

echo "NORMAL GUN VIOLENCE 0, 2"
python main_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 0 --num_republican_agents 2 --instruction_file ./files/gun_violence.txt

echo "NORMAL GUN VIOLENCE 1, 1"
python main_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 1 --num_republican_agents 1 --instruction_file ./files/gun_violence.txt

# 2つ目のコマンドの実行
echo "NORMAL CLIMATE CHANGE 2, 0"
python main_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 2 --num_republican_agents 0 --instruction_file ./files/climate_change.txt

echo "NORMAL CLIMATE CHANGE 0, 2"
python main_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 0 --num_republican_agents 2 --instruction_file ./files/climate_change.txt

echo "NORMAL CLIMATE CHANGE 1, 1"
python main_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 1 --num_republican_agents 1 --instruction_file ./files/climate_change.txt

echo "ECHO RACISM 2, 0"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 2 --num_republican_agents 0 --instruction_file ./files/echo_racism.txt

echo "ECHO RACISM 0, 2"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 0 --num_republican_agents 2 --instruction_file ./files/echo_racism.txt
echo "ECHO RACISM 1, 1"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 1 --num_republican_agents 1 --instruction_file ./files/echo_racism.txt

# 1つ目のコマンドの実行
echo "ECHO ILLEGAL IMMIGRATION 2, 0"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 2 --num_republican_agents 0 --instruction_file ./files/echo_illegal_immigration.txt

echo "ECHO ILLEGAL IMMIGRATION 0, 2"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 0 --num_republican_agents 2 --instruction_file ./files/echo_illegal_immigration.txt
echo "ECHO ILLEGAL IMMIGRATION 1, 1"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 1 --num_republican_agents 1 --instruction_file ./files/echo_illegal_immigration.txt

# 2つ目のコマンドの実行
echo "ECHO GUN VIOLENCE 2, 0"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 2 --num_republican_agents 0 --instruction_file ./files/echo_gun_violence.txt

echo "ECHO GUN VIOLENCE 0, 2"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 0 --num_republican_agents 2 --instruction_file ./files/echo_gun_violence.txt

echo "ECHO GUN VIOLENCE 1, 1"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 1 --num_republican_agents 1 --instruction_file ./files/echo_gun_violence.txt

# 2つ目のコマンドの実行
echo "ECHO CLIMATE CHANGE 2, 0"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 2 --num_republican_agents 0 --instruction_file ./files/echo_climate_change.txt

echo "ECHO CLIMATE CHANGE 0, 2"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 0 --num_republican_agents 2 --instruction_file ./files/echo_climate_change.txt

echo "ECHO CLIMATE CHANGE 1, 1"
python main_echo_batch.py --trial_times 40 --round_robin_times 10 --num_democrat_agents 1 --num_republican_agents 1 --instruction_file ./files/echo_climate_change.txt
# 終了メッセージを表示
echo "Batch scripts completed."
