import sys
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm  # tqdmで進捗バーを表示

def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_personas.py <prompt_file.txt>")
        sys.exit(1)

    prompt_file = sys.argv[1]
    output_file = prompt_file.rsplit('.', 1)[0] + '.csv'

    # プロンプトファイルの読み込み
    with open(prompt_file, 'r', encoding='utf-8') as file:
        prompt = file.read()

    # トークナイザとモデルの読み込み
    model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 生成パラメータの設定
    generation_kwargs = {
        'max_length': 512,           # 必要に応じて調整してください
        'temperature': 1.0,
        'do_sample': True,
        'top_p': 0.95,
        'top_k': 50,
        'num_return_sequences': 1,
        'pad_token_id': tokenizer.eos_token_id
    }

    # プロンプトのトークン化
    prompt_tokens = tokenizer(prompt, return_tensors='pt').to(device)
    prompt_length = prompt_tokens.input_ids.shape[1]

    # 50個のペルソナを生成
    personas = []
    for i in tqdm(range(50), desc="Generating personas", unit="persona"):
        with torch.no_grad():
            outputs = model.generate(**prompt_tokens, **generation_kwargs)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # プロンプト部分を削除
        persona = generated_text[len(prompt):].strip()
        personas.append(persona)

    # CSVファイルに保存
    with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Persona'])
        for persona in personas:
            writer.writerow([persona])

    print(f"Generated personas saved to {output_file}")

if __name__ == '__main__':
    main()

