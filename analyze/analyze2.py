import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np

def main():
    # parser settiong
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_file', type=str, required=True, help='データ分析対象のファイル')
    parser.add_argument('--save_name', type=str, required=True, help='保存するときの名前')
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.target_file)

    # ラウンドを数える
    rounds = sorted(df['Round'].unique())
    sessions = sorted(df['Session'].unique())

    # partyリストを取得する。
    party_list = df[(df['Session'] == 1) & (df['Round'] == 0)]['Party'].tolist()
    print('party_list')
    print(party_list)
    print()

    # グラフの準備
    plt.figure(figsize=(10,6))

    # パーティの数だけ実施する。
    used_names = set()
    for party in party_list:
        print(f"\nNow Party is {party}.")
        agent_names = []
        # 各セッションから該当パーティのエージェントを一人選出する。
        for session in range(1, len(sessions)+1):
            candidates = df[(df['Session'] == session) & (df['Party'] == party)]['Agent Name'].unique()
            agent_name = next((name for name in candidates if name not in used_names), None)
            print(f"From session {session}, extracted agent_name is {agent_name}.")
            
            if agent_name:
                agent_names.append(agent_name)
                used_names.add(agent_name)
            else:
                raise ValueError(f"No available agent found for Party: {party} in Session: {session}")

        # このパーティのエージェントの意見の平均とエラーバーの推移を求める。
        mean_responses = []
        std_responses = []

        for round_num in rounds:
            # 該当するエージェントかつ該当するラウンドのエージェントdfを取得
            filtered_df = df[(df['Agent Name'].isin(agent_names)) & (df['Round'] == round_num)]
            print(f"Round {round_num}")
            print(filtered_df)
            print()

            mean_response = filtered_df['Response'].mean()
            std_response = filtered_df['Response'].std()

            mean_responses.append(mean_response)
            std_responses.append(std_response)

        # グラフにプロット
        if party == 'Democrat':
            color = 'blue'
        elif party == 'Republican':
            color = 'red'
        else:
            color = 'green'
        
        plt.errorbar(rounds, mean_responses, yerr=std_responses, label=party, color=color, capsize=5, marker='o', linestyle='-')

    
    # グラフの装飾
    plt.xlabel('Rounds')
    plt.ylabel('Attitude Score')
    plt.ylim(0,10)
    plt.title(args.save_name)
    plt.legend(title='Party')

    # グラフの保存
    plt.savefig(args.save_name)


if __name__ == '__main__':
    main()
