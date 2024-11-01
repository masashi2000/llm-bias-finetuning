import random
import transformers
import torch
import csv
import os
import sys
import argparse
from datetime import datetime

class Agent:
    def __init__(self, name, persona, party):
        self.name = name
        self.persona = f"Your name is {name}.\n{persona}"
        self.party = party

    def generate_response(self, instruction_and_history, generator):
        prompt = [
            {"role": "system", "content": self.persona},
            {"role": "user", "content": instruction_and_history}
        ]
        generation = generator(
            prompt,
            temperature=1.0,
            top_p=1,
            max_new_tokens=512,
            pad_token_id = generator.model.config.eos_token_id[0]
        )
        generated_text = generation[0]['generated_text'][-1]["content"]
        try:
            assistant_reply = generated_text.split('assistant')[-1].split('content')[-1].strip(" ':}{\n")
        except IndexError:
            assistant_reply = generated_text
        return assistant_reply

    def generate_survey_response(self, question, conversation_history, generator):
        import re
        max_retries = 10
        for attempt in range(max_retries):
            prompt = [
                {"role": "system", "content": self.persona},
                {"role": "user", "content": conversation_history + question}
            ]
            generation = generator(
                prompt,
                temperature=1.0,
                top_p=1,
                max_new_tokens=50,
                pad_token_id = generator.model.config.eos_token_id[0]
            )
            generated_text = generation[0]['generated_text'][-1]["content"]
            # 数値を抽出
            numbers = re.findall(r'\d+', generated_text)
            if numbers:
                response = int(numbers[0])
                return response
        # 最大リトライ回数に達しても数値が得られなかった場合
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

    def ask_agents_question(self, round_number, generator):
        for agent in self.agents:
            question = f"Hi {agent.name}. {self.survey_question}"
            response = agent.generate_survey_response(question, self.conversation_history, generator)
            self.survey_results.append({
                "Session": self.session_number,
                "Round": round_number,
                "Agent Name": agent.name,
                "Party": agent.party,
                "Response": response
            })

    def run(self, generator):
        # 会話開始前の質問
        self.ask_agents_question(round_number=0, generator=generator)
        for round_num in range(1, self.round_robin_times + 1):
            agents_order = self.agents[:]
            # random.shuffle(agents_order)
            for agent in agents_order:
                instruction_and_history = f"{self.instruction}\n{self.conversation_history}"
                agent_response = agent.generate_response(instruction_and_history, generator)
                self.conversation_history += f"{agent.name}: {agent_response}\n"
                yield {
                    "Session": self.session_number,
                    "Round": round_num,
                    "Agent Name": agent.name,
                    "Party": agent.party,
                    "Response": agent_response
                }
            # ラウンド終了後の質問
            self.ask_agents_question(round_number=round_num, generator=generator)

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
        # 自分が後からつけた
        session_counter = 0

        # 試行回数分の試行をする。
        for session_number in range(1, self.trial_times + 1):
            session_counter += 1
            print()
            print("Now on session {}.".format(session_counter))
            print()
            
            #セッションに使う名前の取得
            required_names = self.num_democrat_agents + self.num_republican_agents
            available_names = [name for name in self.names_list if name not in self.used_names]
            if len(available_names) < required_names:
                raise ValueError("利用可能な名前が不足しています。")

            session_names = random.sample(available_names, required_names)
            self.used_names.update(session_names)

            # ペルソナの取得
            available_democrat_personas = [p for p in self.democrat_personas_list if p not in self.used_democrat_personas]
            available_republican_personas = [p for p in self.republican_personas_list if p not in self.used_republican_personas]

            if len(available_democrat_personas) < self.num_democrat_agents:
                raise ValueError("利用可能な民主党ペルソナが不足しています。")

            if len(available_republican_personas) < self.num_republican_agents:
                raise ValueError("利用可能な共和党ペルソナが不足しています。")

            # 名前とペルソナをそれぞれの政党にアサイン
            session_democrat_names = session_names[:self.num_democrat_agents]
            session_republican_names = session_names[self.num_democrat_agents:]

            session_democrat_personas = random.sample(available_democrat_personas, self.num_democrat_agents)
            session_republican_personas = random.sample(available_republican_personas, self.num_republican_agents)

            self.used_democrat_personas.update(session_democrat_personas)
            self.used_republican_personas.update(session_republican_personas)

            # エージェントインスタンスの生成
            agents = []
            for name, persona in zip(session_democrat_names, session_democrat_personas):
                agent = Agent(name, persona, "Democrat")
                agents.append(agent)

            for name, persona in zip(session_republican_names, session_republican_personas):
                agent = Agent(name, persona, "Republican")
                agents.append(agent)

            # セッションの生成
            session = Session(
                session_number,
                self.round_robin_times,
                agents,
                self.instruction,
                self.survey_question
            )

            conversation_records = []
            for conversation in session.run(generator):
                conversation_records.append(conversation)

            # 会話履歴をCSVに保存
            conversation_file = os.path.join(output_dir, f"session_{session_number}_conversation.csv")
            with open(conversation_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ["Session", "Round", "Agent Name", "Party", "Response"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for record in conversation_records:
                    writer.writerow(record)

            # 質問結果を集計
            all_survey_results.extend(session.survey_results)

        # 質問結果をCSVに保存
        survey_file = os.path.join(output_dir, "survey_results.csv")
        with open(survey_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["Session", "Round", "Agent Name", "Party", "Response"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in all_survey_results:
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
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    available_device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = transformers.pipeline(
        "text-generation",
        model=model_id,
        device=available_device,
        torch_dtype=torch.bfloat16
    )
    # これがないと並列処理ができない。
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
    output_dir = f"{timestamp}_{instruction_filename}_Dem{args.num_democrat_agents}_Rep{args.num_republican_agents}_Round{args.round_robin_times}_Trial{args.trial_times}"
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
