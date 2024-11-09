#!/bin/bash

# Define log file
LOG_FILE="execution_log.txt"

# First command
echo "Batch 3 , 1, 1" | tee -a $LOG_FILE
python main_batch_3.py --trial_times 40 --round_robin_times 20 --num_democrat_agents 1 --num_republican_agents 1 --instruction_file ./files/racism.txt 2>&1 | tee -a $LOG_FILE

echo "Batch 4, 2, 0" | tee -a $LOG_FILE
python main_batch_4.py --trial_times 40 --round_robin_times 20 --num_democrat_agents 2 --num_republican_agents 0 --instruction_file ./files/racism.txt 2>&1 | tee -a $LOG_FILE

echo "Batch 5, 0, 2" | tee -a $LOG_FILE
python main_batch_5.py --trial_times 40 --round_robin_times 20 --num_democrat_agents 0 --num_republican_agents 2 --instruction_file ./files/racism.txt 2>&1 | tee -a $LOG_FILE

echo "Batch 6, 1, 1" | tee -a $LOG_FILE
python main_batch_6.py --trial_times 40 --round_robin_times 20 --num_democrat_agents 1 --num_republican_agents 1 --instruction_file ./files/racism.txt 2>&1 | tee -a $LOG_FILE

echo "All commands executed. Check $LOG_FILE for details."


