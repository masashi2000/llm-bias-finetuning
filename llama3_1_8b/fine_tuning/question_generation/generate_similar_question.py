import transformers
import torch
import yaml
import csv
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='Path to the model config file')
    args = parser.parse_args()

    # Load config file
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    # Initialize model
    model_id = config["model_config"]["model_id"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        "Could you discuss your perspective on significant political issues facing America today?", 
        "How do you balance Second Amendment rights with the need for gun control measures?",
        "How do you balance the need for national security with the preservation of personal freedoms?", 
        "How do you believe the U.S. should handle immigration and border security?",
        "What core political ideals most significantly shape your viewpoint on governance and policy-making?",
        "What are your views on racial inequality and systemic racism in American society?",
        "What is your stance on the government's role in addressing climate change and environmental protection?",
        "What role do you think diversity plays in shaping the cultural landscape of America?", 
        "What values do you believe are essential to the American identity?", 
        "Which political issues do you believe are most urgent for the next president to address?",
    ]
    # Generate questions.
    GENERATE_QUESTION_TEMPLATE = """
    Rephrase the question below:
    {question}
    You must follow the format below:
    -----------------------------------
    Question : Rephrased_Question"""

    user_prompts = [GENERATE_QUESTION_TEMPLATE.format(question=q) for q in questions]

    # プロンプトの準備
    prompts = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ]
        for user_prompt in user_prompts
    ]

    # CSVファイルに書き出す準備
    with open('questions.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # ヘッダー行を書き込む
        writer.writerow(['Original Question', 'Rephrased Question'])

        # 各プロンプトに対してリフレーズされた質問を生成
        for original_question, prompt in zip(questions, prompts):
            for _ in range(10):
                max_attempts = 3  # 抽出を試みる最大回数
                attempt = 0
                rephrased_question = None
                while attempt < max_attempts:
                    output = generator(
                        prompt,
                        do_sample=True,
                        max_new_tokens=300,
                        top_p=1,
                        temperature=1.0,
                        pad_token_id=generator.tokenizer.pad_token_id,
                    )
                    # 生成されたテキストからリフレーズされた質問を抽出
                    generated_text = output[0]['generated_text'][-1]["content"]
                    # 'Question :'から始まる行を探す
                    lines = generated_text.strip().split('\n')
                    for line in lines:
                        if line.strip().startswith('Question :'):
                            rephrased_question = line.split('Question :', 1)[1].strip()
                            break
                    if rephrased_question:
                        print()
                        print(rephrased_question)
                        # 抽出に成功した場合、ループを抜ける
                        break
                    else:
                        # 抽出に失敗した場合、試行回数を増やして再度プロンプトを投げる
                        attempt += 1
                if rephrased_question:
                    # オリジナルとリフレーズされた質問をCSVに書き込む
                    writer.writerow([original_question, rephrased_question])
                else:
                    # すべての試行で抽出に失敗した場合、エラーメッセージをCSVに書き込む
                    writer.writerow([original_question, "Failed to generate rephrased question"])

if __name__ == "__main__":
    main()
