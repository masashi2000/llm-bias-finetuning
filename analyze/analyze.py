import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--target_file', type=str, required=True, help='データ分析に使うファイル')
parser.add_argument('--save_name', type=str, required=True, help='保存するときの名前')
args = parser.parse_args()

# Load data
data = pd.read_csv(args.target_file)

# Group data by Round and Session, then separate into two sets (A and B) as per instructions
rounds = sorted(data['Round'].unique())
sessions = sorted(data['Session'].unique())

# To store mean responses and standard errors for A and B at each round
mean_A, mean_B = [], []
error_A, error_B = [], []

for r in rounds:
    # Filter rows for the current round
    round_data = data[data['Round'] == r]
    
    # For each session, separate the first and second response (A and B)
    responses_A = round_data.groupby('Session').nth(0)['Response']
    responses_B = round_data.groupby('Session').nth(1)['Response']
    
    # Calculate means and standard errors for A and B
    mean_A.append(responses_A.mean())
    mean_B.append(responses_B.mean())
    error_A.append(responses_A.std() / np.sqrt(len(responses_A)))
    error_B.append(responses_B.std() / np.sqrt(len(responses_B)))

# Plotting
plt.figure(figsize=(8, 6))
plt.errorbar(rounds, mean_A, yerr=error_A, label='A', color='red', fmt='-o', capsize=5)
plt.errorbar(rounds, mean_B, yerr=error_B, label='B', color='blue', fmt='-o', capsize=5)
plt.axhline(y=8, color='orange', linestyle='--', label='Threshold')

# Adding labels and legend
plt.xlabel('Round')
plt.ylabel('Average Response')
plt.title('Average Response Over Rounds by Agent Groups A and B')
plt.legend()
plt.savefig(args.save_name+'.png')

