import time
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

# モデル設定
model_kwargs = {
    "device_map": "auto",
    "torch_dtype": torch.bfloat16
}

pipeline_kwargs = {
    "do_sample": False,
    "max_new_tokens": 800,
}

# モデルとトークナイザーのロード
checkpoint = "gpt2"  # 必要に応じてモデル名を変更
model = AutoModelForCausalLM.from_pretrained(checkpoint, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# テキスト生成パイプラインの作成
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=pipeline_kwargs["max_new_tokens"]
)

# パディングトークン設定
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

# 入力テキストリスト
input_texts = [
    "Once upon a time in a faraway land,",
    "In the future, technology will revolutionize",
    "The mystery of the ancient ruins was finally solved",
    "In a small village, a legend was born",
    "Exploring the depths of the ocean, scientists discovered"
]

# 各バッチサイズでの処理速度を計測
batch_times = {}
for batch_size in range(1, 6):
    llm = HuggingFacePipeline(
        pipeline=pipe,
        model_kwargs=model_kwargs,
        pipeline_kwargs=pipeline_kwargs,
        batch_size=batch_size
    )

    start_time = time.time()
    outputs = llm.generate(input_texts)  # バッチ処理でテキスト生成
    end_time = time.time()

    batch_times[batch_size] = end_time - start_time  # 各バッチサイズの処理時間を記録

# 結果の表示
for batch_size, exec_time in batch_times.items():
    print(f"Batch size {batch_size}: {exec_time:.2f} seconds")

