import random
import transformers
import torch
import csv
import os
import sys
import argparse
from datetime import datetime
from tqdm import tqdm

class Agent:
    def __init__(self, name, persona, party):
        self.name = name
        self.persona = f"Your name is {name}.\n{persona}"
        self.party = party

class Session:
    def __init__(self, session_number, round_robin_times, agents, instruction, survey_question):
        self.session_number = session_number
        self.round_robin_times = round_robin_times
        self.instruction = instruction
        self.agents = agents
        self.survey_question = survey_question
        self.conversation_history = ""
        self.survey_results = []

    def initialize_agents(self):
        for agent in self.agents:
            agent.conversation_history = ""

    def get_agent_prompts(self):
        prompts = []
        for agent in self.agents:
            instruction_and_history = f"{self.instruction}\n{agent.conversation_history}"
            prompt = f"{agent.persona}\n{instruction_and_history}"
            prompts.append(prompt)
        return prompts

    def update_agent_histories(self, responses):
        for agent, response in zip(self.agents, responses):
            agent.conversation_history += f"{agent.name}: {response}\n"
            self.conversation_history += f"{agent.name}: {response}\n"

    def get_survey_prompts(self, round_number):
        prompts = []
        for agent in self.agents:
            question = f"Hi {agent.name}. {self.survey_question}"
            prompt = f"{agent.persona}\n{agent.conversation_history}\n{question}"
            prompts.append(prompt)
        return prompts

    def update_survey_results(self, responses, round_number):
        import re
        for agent, response in zip(self.agents, responses):
            numbers = re.findall(r'\d+', response)
            if numbers:
                survey_response = int(numbers[0])
            else:
                survey_response = -1
            self.survey_results.append({
                "Session": self.session_number,
                "Round": round_number,
                "Agent Name": agent.name,
                "Party": agent.party,
                "Response": survey_response
            })

    def run(self, generator):
        self.initialize_agents()
        # 会話開始前の質問
        survey_prompts = self.get_survey_prompts(round_number=0)
        survey_responses = generator(
            survey_prompts,
            temperature=1.0,
            top_p=1,
            max_new_tokens=50,
            pad_token_id=generator.model.config.eos_token_id[2]
        )
        survey_texts = [response['generated_text'] for response in survey_responses]
        self.update_survey_results(survey_texts, round_number=0)

        for round_num in tqdm(range(1, self.round_robin_times + 1), desc=f"Session {self.session_number} Rounds", leave=False):
            prompts = self.get_agent_prompts()
            responses = generator(
                prompts,
                temperature=1.0,
                top_p=1,
                max_new_tokens=512,
                pad_token_id=generator.model.config.eos_token_id[2]
            )
            response_texts = [response['generated_text'] for response in responses]
            self.update_agent_histories(response_texts)

            for agent, response_text in zip(self.agents, response_texts):
                yield {
                    "Session": self.session_number,
                    "Round": round_num,
                    "Agent Name": agent.name,
                    "Party": agent.party,
                    "Response": response_text
                }

            # ラウンド終了後の質問
            survey_prompts = self.get_survey_prompts(round_number=round_num)
            survey_responses = generator(
                survey_prompts,
                temperature=1.0,
                top_p=1,
                max_new_tokens=50,
                pad_token_id=generator.model.config.eos_token_id[2]
            )
            survey_texts = [response['generated_text'] for response in survey_responses]
            self.update_survey_results(survey_texts, round_number=round_num)

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
        self.used_names = set()
        self.used_democrat_personas = set()
        self.used_republican_personas = set()
        self.instruction = instruction
        self.survey_question = survey_question

    def run(self, generator, output_dir):
        all_survey_results = []
        session_counter = 0
        sessions = []
        for session_number in tqdm(range(1, self.trial_times + 1), desc="Sessions"):
            session_counter += 1
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

            democrat_names = session_names[:self.num_democrat_agents]
            republican_names = session_names[self.num_democrat_agents:]

            session_democrat_personas = random.sample(available_democrat_personas, self.num_democrat_agents)
            session_republican_personas = random.sample(available_republican_personas, self.num_republican_agents)

            self.used_democrat_personas.update(session_democrat_personas)
            self.used_republican_personas.update(session_republican_personas)

            agents = []
            for name, persona in zip(democrat_names, session_democrat_personas):
                agent = Agent(name, persona, "Democrat")
                agents.append(agent)

            for name, persona in zip(republican_names, session_republican_personas):
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

        # セッションをバッチ処理
        all_conversation_records = []
        for round_num in tqdm(range(self.round_robin_times + 1), desc="Processing Rounds"):
            # 各セッションからプロンプトを収集
            batch_prompts = []
            batch_sessions = []
            for session in sessions:
                if round_num == 0:
                    # 会話開始前の質問
                    survey_prompts = session.get_survey_prompts(round_number=0)
                    batch_prompts.extend(survey_prompts)
                    batch_sessions.extend([(session, "survey", 0)] * len(survey_prompts))
                else:
                    # 会話
                    prompts = session.get_agent_prompts()
                    batch_prompts.extend(prompts)
                    batch_sessions.extend([(session, "conversation", round_num)] * len(prompts))
            # モデルにバッチ入力
            batch_responses = generator(
                batch_prompts,
                temperature=1.0,
                top_p=1,
                max_new_tokens=512,
                pad_token_id=generator.model.config.eos_token_id[2]
            )
            # 応答を各セッションに振り分け
            for (session, task_type, task_round), response in zip(batch_sessions, batch_responses):
                response_text = response['generated_text']
                if task_type == "survey":
                    session.update_survey_results([response_text], round_number=task_round)
                elif task_type == "conversation":
                    session.update_agent_histories([response_text])
                    agent = session.agents.pop(0)
                    all_conversation_records.append({
                        "Session": session.session_number,
                        "Round": task_round,
                        "Agent Name": agent.name,
                        "Party": agent.party,
                        "Response": response_text
                    })
            # ラウンド終了後の質問
            if round_num > 0:
                batch_prompts = []
                batch_sessions = []
                for session in sessions:
                    survey_prompts = session.get_survey_prompts(round_number=round_num)
                    batch_prompts.extend(survey_prompts)
                    batch_sessions.extend([(session, "survey", round_num)] * len(survey_prompts))
                batch_responses = generator(
                    batch_prompts,
                    temperature=1.0,
                    top_p=1,
                    max_new_tokens=50,
                    pad_token_id=generator.model.config.eos_token_id[2]
                )
                for (session, _, task_round), response in zip(batch_sessions, batch_responses):
                    response_text = response['generated_text']
                    session.update_survey_results([response_text], round_number=task_round)

        # 会話履歴をCSVに保存
        conversation_file = os.path.join(output_dir, f"conversation_records.csv")
        with open(conversation_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["Session", "Round", "Agent Name", "Party", "Response"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for record in all_conversation_records:
                writer.writerow(record)

        # 質問結果をCSVに保存
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
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    available_device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = transformers.pipeline(
        "text-generation",
        model=model_id,
        device=0 if available_device == "cuda" else -1,
        torch_dtype=torch.bfloat16
    )
    # バッチ処理のためにはここでpad token idを変更しておく必要があるみたい。
    generator.tokenizer.pad_token_id = generator.model.config.eos_token_id[0]

    # インストラクションの読み込み
    with open(args.instruction_file, 'r', encoding='utf-8') as f:
        instruction_content = f.read()
    # 質問とインストラクションを分離（質問はインストラクションの最後の行にあると仮定）
    instruction_lines = instruction_content.strip().split('\n')
    instruction = '\n'.join(instruction_lines[:-1])
    survey_question = instruction_lines[-1]

    # CSVファイルからリストを読み込む
    names_list = []
    with open('names.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            names_list.append(row['name'])

    democrat_personas_list = []
    with open('prompt_file_democrat_v2.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            democrat_personas_list.append(row['Persona'])

    republican_personas_list = []
    with open('prompt_file_republican_v2.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            republican_personas_list.append(row['Persona'])

    # 出力ディレクトリの作成
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    instruction_filename = os.path.basename(args.instruction_file).split('.')[0]
    output_dir = f"{instruction_filename}_Dem{args.num_democrat_agents}_Rep{args.num_republican_agents}_Trial{args.trial_times}_Round{args.round_robin_times}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

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

