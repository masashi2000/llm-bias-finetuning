# 実行するときのフォルダの指定などに注意
# python persona_generation/generate_persona.py  --prompt_file persona_generation/prompt_file_democrat_v2.txt --config_file ./llama3_1_8b/config.yml
import transformers
import torch
import sys
import csv
import os
import yaml
import argparse



# 引数の読み込み
parser = argparse.ArgumentParser()
parser.add_argument('--prompt_file', type=str, required=True, help='プロンプトのファイル')
parser.add_argument('--config_file', type=str, required=True, help='モデルの設定ファイルが必要')
args = parser.parse_args()

# Read the prompt from the specified file
with open(args.prompt_file, 'r', encoding='utf-8') as f:
    prompt_text = f.read()

# Configファイルの読み込み
with open(args.config_file, "r") as file:
    config = yaml.safe_load(file)

# Initialize the pipeline with the specified settings
model_id = config["model_config"]["model_id"]
avairable_device = "cuda" if torch.cuda.is_available() else "cpu"

generator = transformers.pipeline(
    model=model_id,
    device=avairable_device,
    torch_dtype=torch.bfloat16
)


# Generate 50 personas
personas = []

prompt = [{"role": "system", "content": "You are a helpful assistant."}]
prompt.append({"role": "user", "content": prompt_text})

for i in range(100):
    generation = generator(
        prompt,
        temperature=1.0,
        top_p=1,
        max_new_tokens=512,
        do_sample=True,
    )
    personas.append(generation[0]["generated_text"][2]["content"])
    print("{} : {}".format(i+1, generation[0]["generated_text"][2]["content"]))
    print()

# Prepare the CSV file name based on the prompt file name
base_name = os.path.splitext(os.path.basename(args.prompt_file))[0]
csv_file = base_name + '.csv'

# Write the generated personas to a CSV file
with open(csv_file, 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'Persona'])
    for i, persona in enumerate(personas, 1):
        writer.writerow([i, persona])

