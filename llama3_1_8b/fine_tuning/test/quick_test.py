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

    with open("persona_example.txt", "r") as file:
        persona = file.read()

    with open("instruction_example.txt", "r") as file:
        instruction = file.read()

    prompt = [
            {"role": "system", "content": persona},
            {"role": "user", "content": instruction}
    ]


    for i in range(1,50):
        output = generator(
            prompt,
            do_sample=True,
            max_new_tokens=512,
            top_p=1,
            temperature=1.0,
            pad_token_id=generator.tokenizer.pad_token_id,
        )
        # 生成されたテキストからリフレーズされた質問を抽出
        generated_text = output[0]['generated_text'][-1]["content"]
        print(f"{i} : {generated_text}")

if __name__ == "__main__":
    main()
