import transformers
import torch
import yaml

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

    questions = [
        "What do you think about illegal immigration?",
        "What do you think about racism?",
        "What do you think about gun violence?",
        "What do you think about climate change?",
    ]

    # Generate questions.
    GENERATE_QUESTION_TEMPLATE = """
    {questin}
    Answer shorter than 50 words."""

    user_prompts = [GENERATE_QUESTION_TEMPLATE.format(question=q) for q in questions]

    # ペルソナの準備
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

    # プロンプトの準備

    prompts = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ]
        for user_prompt in user_prompts
    ]


    for prompt in prompts:
        output = generator(
                prompt,
                do_sample=True,
                max_new_tokens=300,
                top_p=1,
                temperature=1.0,
                pad_token_id=generator.tokenizer.pad_token_id,
                )
        print()
        print(output[0]['generated_text'][-1]["content"])


if __name__ == "__main__":
    main()
