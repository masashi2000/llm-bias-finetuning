import torch
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

llama_31 = "meta-llama/Llama-3.1-8B-Instruct"  # <-- llama 3.1
llama_32 = "meta-llama/Llama-3.2-3B-Instruct"  # <-- llama 3.2

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

# モデルのパイプライン設定 (batch_size=3に設定してバッチ処理を実施)
generator = pipeline(model=llama_31, device=device, torch_dtype=torch.bfloat16)

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

