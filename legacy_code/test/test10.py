import time
from datasets import Dataset
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# 入力テキストのリスト
input_texts = [
    "Once upon a time in a faraway land,",
    "In the future, technology will revolutionize",
    "The mystery of the ancient ruins was finally solved",
    "In a small village, a legend was born",
    "Exploring the depths of the ocean, scientists discovered"
]

# データセットの作成
dataset = Dataset.from_dict({"text": input_texts})

# モデルとトークナイザーのロード
checkpoint = "gpt2"  # 使用したいモデルに合わせて変更してください
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# `pad_token_id`を設定
tokenizer.pad_token_id = tokenizer.eos_token_id

# 処理時間を記録する辞書
batch_times = {}

# 各バッチサイズでの処理時間を計測
for batch_size in range(1, 6):
    # パイプラインの作成
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,  # 出力の最大トークン数を指定
        batch_size=batch_size
    )

    # バッチ処理の開始時間を記録
    start_time = time.time()
    
    # バッチ処理の実行
    results = dataset.map(lambda examples: {"generated_text": pipe(examples["text"])}, batched=True, batch_size=batch_size)
    
    # 終了時間を記録
    end_time = time.time()

    # 各バッチサイズの処理時間を辞書に記録
    batch_times[batch_size] = end_time - start_time

# 各バッチサイズの処理時間を表示
for batch_size, exec_time in batch_times.items():
    print(f"Batch size {batch_size}: {exec_time:.2f} seconds")

