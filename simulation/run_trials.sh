#!/bin/bash

# Define log file
LOG_FILE="execution_log.txt"

# First command
echo "Executing trial with 2 Democrat agents and 0 Republican agents..." | tee -a $LOG_FILE
python main.py --trial_times 30 --round_robin_times 20 --num_democrat_agents 2 --num_republican_agents 0 --instruction_file racism.txt 2>&1 | tee -a $LOG_FILE

# Second command
echo "Executing trial with 0 Democrat agents and 2 Republican agents..." | tee -a $LOG_FILE
python main.py --trial_times 30 --round_robin_times 20 --num_democrat_agents 0 --num_republican_agents 2 --instruction_file racism.txt 2>&1 | tee -a $LOG_FILE

# Third command
echo "Executing trial with 1 Democrat agent and 1 Republican agent..." | tee -a $LOG_FILE
python main.py --trial_times 30 --round_robin_times 20 --num_democrat_agents 1 --num_republican_agents 1 --instruction_file racism.txt 2>&1 | tee -a $LOG_FILE

echo "All commands executed. Check $LOG_FILE for details."

