import torch
from transformers import pipeline, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

llama_31 = "meta-llama/Llama-3.1-8B-Instruct"

# バッチ内のプロンプトリスト
prompts = [
    [
        {"role": "system", "content": "You are a helpful assistant, that responds as a pirate."},
        {"role": "user", "content": "What's Deep Learning?"}
    ],
    [
        {"role": "system", "content": "You are a knowledgeable professor."},
        {"role": "user", "content": "Explain quantum computing briefly."}
    ],
    [
        {"role": "system", "content": "You are a playful storyteller."},
        {"role": "user", "content": "Tell me a short story about a brave cat."}
    ]
]

# モデルとトークナイザーの読み込み
generator = pipeline(model=llama_31, device=device, torch_dtype=torch.bfloat16)
tokenizer = generator.tokenizer

# `pad_token_id` を設定
tokenizer.pad_token_id = tokenizer.eos_token_id

# バッチ処理でプロンプトのリストを生成
generation = generator(
    prompts,
    do_sample=False,
    max_new_tokens=50,
    batch_size=2
)

# 結果の表示
for i, output in enumerate(generation):
    print(f"Generation {i+1}: {output['generated_text']}")

