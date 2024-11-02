import random
import transformers
import torch
import csv
import os
import sys
import argparse
from datetime import datetime
import re

class Agent:
    def __init__(self, name, persona, party):
        self.name = name
        self.persona = f"Your name is {name}.\n{persona}"
        self.party = party

    def construct_response_prompt(self, instruction_and_history):
        # プロンプトをリスト形式で構築
        prompt = [
            {"role": "system", "content": self.persona},
            {"role": "user", "content": instruction_and_history}
        ]
        return prompt

    def construct_survey_prompt(self, conversation_history, question):
        prompt = [
            {"role": "system", "content": self.persona},
            {"role": "user", "content": conversation_history + question}
        ]
        return prompt

    def extract_response(self, generated_text):
        # アシスタントの応答を抽出
        return generated_text.strip()

    def extract_number_from_response(self, generated_text):
        numbers = re.findall(r'\d+', generated_text)
        if numbers:
            response = int(numbers[0])
            return response
        else:
            return -1

class Session:
    def __init__(self, session_number, round_robin_times, agents, instruction, survey_question):
        self.session_number = session_number
        self.round_robin_times = round_robin_times
        self.instruction = instruction
        self.agents = agents
        self.survey_question = survey_question
        self.conversation_history = ""
        self.survey_results = []

class Experiment:
    def __init__(self, trial_times, round_robin_times, num_democrat_agents, num_republican_agents,
                 names_list, democrat_personas_list, republican_personas_list, instruction, survey_question):
        self.trial_times = trial_times
        self.round_robin_times = round_robin_times
        self.num_democrat_agents = num_democrat_agents
        self.num_republican_agents = num_republican_agents
        self.names_list = names_list
        self.democrat_personas_list = democrat_personas_list
        self.republican_personas_list = republican_personas_list
        self.instruction = instruction
        self.survey_question = survey_question
        self.used_names = set()
        self.used_democrat_personas = set()
        self.used_republican_personas = set()

    def run(self, generator, output_dir):
        all_survey_results = []
        conversation_records = []
        sessions = []

        # セッションの作成
        for session_number in range(1, self.trial_times + 1):
            # 名前とペルソナの選択（詳細は省略）
            required_names = self.num_democrat_agents + self.num_republican_agents
            available_names = [name for name in self.names_list if name not in self.used_names]
            if len(available_names) < required_names:
                raise ValueError("利用可能な名前が不足しています。")

            session_names = random.sample(available_names, required_names)
            self.used_names.update(session_names)

            available_democrat_personas = [p for p in self.democrat_personas_list if p not in self.used_democrat_personas]
            available_republican_personas = [p for p in self.republican_personas_list if p not in self.used_republican_personas]

            if len(available_democrat_personas) < self.num_democrat_agents:
                raise ValueError("利用可能な民主党ペルソナが不足しています。")

            if len(available_republican_personas) < self.num_republican_agents:
                raise ValueError("利用可能な共和党ペルソナが不足しています。")

            session_democrat_names = session_names[:self.num_democrat_agents]
            session_republican_names = session_names[self.num_democrat_agents:]

            session_democrat_personas = random.sample(available_democrat_personas, self.num_democrat_agents)
            session_republican_personas = random.sample(available_republican_personas, self.num_republican_agents)

            self.used_democrat_personas.update(session_democrat_personas)
            self.used_republican_personas.update(session_republican_personas)

            # エージェントの作成
            agents = []
            for name, persona in zip(session_democrat_names, session_democrat_personas):
                agent = Agent(name, persona, "Democrat")
                agents.append(agent)

            for name, persona in zip(session_republican_names, session_republican_personas):
                agent = Agent(name, persona, "Republican")
                agents.append(agent)

            session = Session(
                session_number,
                self.round_robin_times,
                agents,
                self.instruction,
                self.survey_question
            )

            sessions.append(session)

        # ラウンドごとの処理
        for round_num in range(0, self.round_robin_times + 1):
            # 初回のアンケート
            if round_num == 0:
                survey_prompts = []
                agent_session_info = []
                for session in sessions:
                    for agent in session.agents:
                        question = f"Hi {agent.name}. {session.survey_question}"
                        prompt = agent.construct_survey_prompt(session.conversation_history, question)
                        survey_prompts.append(prompt)
                        agent_session_info.append((agent, session, round_num))

                # バッチ処理
                batch_generations = generator(survey_prompts, temperature=1.0, top_p=1, max_new_tokens=50)

                # 結果の処理
                for i, generation in enumerate(batch_generations):
                    agent, session, round_num = agent_session_info[i]
                    generated_text = generation['generated_text']
                    response = agent.extract_number_from_response(generated_text)
                    session.survey_results.append({
                        "Session": session.session_number,
                        "Round": round_num,
                        "Agent Name": agent.name,
                        "Party": agent.party,
                        "Response": response
                    })
            else:
                # 会話のラウンド
                conversation_prompts = []
                agent_session_info = []
                for session in sessions:
                    agents_order = session.agents[:]
                    # random.shuffle(agents_order)
                    for agent in agents_order:
                        instruction_and_history = f"{session.instruction}\n{session.conversation_history}"
                        prompt = agent.construct_response_prompt(instruction_and_history)
                        conversation_prompts.append(prompt)
                        agent_session_info.append((agent, session, round_num))

                # バッチ処理
                batch_generations = generator(conversation_prompts, temperature=1.0, top_p=1, max_new_tokens=512)

                # 結果の処理と会話履歴の更新
                for i, generation in enumerate(batch_generations):
                    agent, session, round_num = agent_session_info[i]
                    assistant_reply = generation['generated_text']
                    agent_response = agent.extract_response(assistant_reply)
                    session.conversation_history += f"{agent.name}: {agent_response}\n"
                    conversation_records.append({
                        "Session": session.session_number,
                        "Round": round_num,
                        "Agent Name": agent.name,
                        "Party": agent.party,
                        "Response": agent_response
                    })

                # ラウンド終了後のアンケート
                survey_prompts = []
                agent_session_info = []
                for session in sessions:
                    for agent in session.agents:
                        question = f"Hi {agent.name}. {session.survey_question}"
                        prompt = agent.construct_survey_prompt(session.conversation_history, question)
                        survey_prompts.append(prompt)
                        agent_session_info.append((agent, session, round_num))

                # バッチ処理
                batch_generations = generator(survey_prompts, temperature=1.0, top_p=1, max_new_tokens=50)

                # 結果の処理
                for i, generation in enumerate(batch_generations):
                    agent, session, round_num = agent_session_info[i]
                    generated_text = generation['generated_text']
                    response = agent.extract_number_from_response(generated_text)
                    session.survey_results.append({
                        "Session": session.session_number,
                        "Round": round_num,
                        "Agent Name": agent.name,
                        "Party": agent.party,
                        "Response": response
                    })

        # 結果の保存
        conversation_file = os.path.join(output_dir, "conversation_records.csv")
        with open(conversation_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["Session", "Round", "Agent Name", "Party", "Response"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for record in conversation_records:
                writer.writerow(record)

        survey_file = os.path.join(output_dir, "survey_results.csv")
        with open(survey_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["Session", "Round", "Agent Name", "Party", "Response"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for session in sessions:
                for result in session.survey_results:
                    writer.writerow(result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_times', type=int, required=True, help='実験の試行回数')
    parser.add_argument('--round_robin_times', type=int, required=True, help='ラウンドロビンの回数')
    parser.add_argument('--num_democrat_agents', type=int, required=True, help='民主党エージェントの数')
    parser.add_argument('--num_republican_agents', type=int, required=True, help='共和党エージェントの数')
    parser.add_argument('--instruction_file', type=str, required=True, help='インストラクションファイルのパス')
    args = parser.parse_args()

    # モデルの初期化
    model_id = "meta-llama/Llama-2-7b-chat-hf"  # 適切なモデルIDに変更してください
    available_device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = transformers.pipeline(
        "chat",
        model=model_id,
        device=available_device,
        torch_dtype=torch.bfloat16
    )

    # インストラクションの読み込み
    with open(args.instruction_file, 'r', encoding='utf-8') as f:
        instruction_content = f.read()
    instruction_lines = instruction_content.strip().split('\n')
    instruction = '\n'.join(instruction_lines[:-1])
    survey_question = instruction_lines[-1]

    # CSVファイルからリストを読み込む（省略）

    # 出力ディレクトリの作成（省略）

    experiment = Experiment(
        args.trial_times,
        args.round_robin_times,
        args.num_democrat_agents,
        args.num_republican_agents,
        names_list,
        democrat_personas_list,
        republican_personas_list,
        instruction,
        survey_question
    )

    experiment.run(generator, output_dir)

if __name__ == "__main__":
    main()

