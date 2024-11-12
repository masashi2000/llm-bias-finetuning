#!/bin/bash

# Define log file
LOG_FILE="execution_log.txt"

# First command
echo "trial 40, demo 2, rep 0" | tee -a $LOG_FILE
python ./simulation/main4.py --trial_times 40 --round_robin_times 30 --num_democrat_agents 2 --num_republican_agents 0 --instruction_file ./simulation/racism.txt 2>&1 | tee -a $LOG_FILE

# Second command
echo "Executing trial with 0 Democrat agents and 2 Republican agents..." | tee -a $LOG_FILE
python ./simulation/main4.py --trial_times 40 --round_robin_times 30 --num_democrat_agents 0 --num_republican_agents 2 --instruction_file ./simulation/racism.txt 2>&1 | tee -a $LOG_FILE

# Third command
#echo "Executing trial with 1 Democrat agent and 1 Republican agent..." | tee -a $LOG_FILE
#python ./echo_chamber/main4.py --trial_times 40 --round_robin_times 30 --num_democrat_agents 1 --num_republican_agents 1 --instruction_file ./echo_chamber/echo_racism.txt 2>&1 | tee -a $LOG_FILE

# Third command
echo "Executing trial with 1 Democrat agent and 1 Republican agent..." | tee -a $LOG_FILE
python ./echo_chamber/main4.py --trial_times 40 --round_robin_times 30 --num_democrat_agents 2 --num_republican_agents 0 --instruction_file ./echo_chamber/echo_racism.txt 2>&1 | tee -a $LOG_FILE

# Third command
echo "Executing trial with 1 Democrat agent and 1 Republican agent..." | tee -a $LOG_FILE
python ./echo_chamber/main4.py --trial_times 40 --round_robin_times 30 --num_democrat_agents 0 --num_republican_agents 2 --instruction_file ./echo_chamber/echo_racism.txt 2>&1 | tee -a $LOG_FILE

echo "All commands executed. Check $LOG_FILE for details."


