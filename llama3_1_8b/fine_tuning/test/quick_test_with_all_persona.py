import transformers
import torch
import yaml
import csv

def main():
    # 設定ファイルのロード
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    # モデルの初期化
    model_id = config["model_config"]["model_id"]
    available_device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = transformers.pipeline(
        "text-generation",
        model=model_id,
        device=available_device,
        torch_dtype=torch.bfloat16
    )

    # これがないとバッチ処理がうまくいかない(モデル別で処理）
    if model_id == "meta-llama/Llama-3.1-8B-Instruct":
        generator.tokenizer.pad_token_id = generator.model.config.eos_token_id[0]
        generator.tokenizer.padding_side = 'left'
    elif model_id == "mistralai/Mistral-7B-Instruct-v0.3":
        generator.tokenizer.pad_token_id = generator.model.config.eos_token_id
    else:
       raise ValueError("モデルの設定がありません。追加してください。")

    # プロンプトの準備
    democrat_personas_list = []
    with open('files/prompt_file_democrat_v2.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            democrat_personas_list.append(row['Persona'])

    republican_personas_list = []
    with open('files/prompt_file_republican_v2.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            republican_personas_list.append(row['Persona'])


    with open("instruction_example.txt", "r") as file:
        instruction = file.read()



    for i in range(1,50):
        persona = democrat_personas_list.pop()
        persona = "You are passionate Democrat. \nYou believe illegal immigration is not a problem.\nYou believe gun violence is a big problem.\nYou believe racism is a big problem.\nYou believe climate change is a big problem."
        persona = ""
        prompt = [
                {"role": "system", "content": persona},
                {"role": "user", "content": instruction}
        ]
        output = generator(
            prompt,
            do_sample=True,
            max_new_tokens=512,
            top_p=1,
            temperature=1.0,
            pad_token_id=generator.tokenizer.pad_token_id,
        )
        generated_text = output[0]['generated_text'][-1]["content"]
        print("---------------------------------")
        print("Democrat")
        print("Persona is as below:")
        print(persona)
        print()
        print("Generated_text is below:")
        print(generated_text)

        persona = republican_personas_list.pop()
        persona = "You are passionate Republican. \nYou believe illegal immigration is a big problem.\nYou believe gun violence is not a problem.\nYou believe racism is not a problem.\nYou believe climate change is not a problem."
        persona = "You are a helphul assistant."
        prompt = [
                {"role": "system", "content": persona},
                {"role": "user", "content": instruction}
        ]
        output = generator(
            prompt,
            do_sample=True,
            max_new_tokens=512,
            top_p=1,
            temperature=1.0,
            pad_token_id=generator.tokenizer.pad_token_id,
        )
        generated_text = output[0]['generated_text'][-1]["content"]
        print("---------------------------------")
        print("Republican")
        print("Persona is as below:")
        print(persona)
        print()
        print("Generated_text is below:")
        print(generated_text)

if __name__ == "__main__":
    main()
