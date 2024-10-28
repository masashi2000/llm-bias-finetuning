import transformers
import torch
import time

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Tokenizerとモデルパイプラインのロード
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

# タイミングの計測開始
start_time = time.time()

# レスポンスの生成
outputs = pipeline(
    messages,
    max_new_tokens=256,
)

# タイミングの計測終了
end_time = time.time()

# 経過時間とトークン毎秒の計算
elapsed_time = end_time - start_time
generated_text = outputs[0]["generated_text"] if isinstance(outputs, list) else outputs

# 生成されたテキストが文字列であることを確認
if not isinstance(generated_text, str):
    generated_text = str(generated_text)

# 生成されたテキストをトークナイズして正確なトークン数を取得
tokens = tokenizer(generated_text, return_tensors="pt")["input_ids"]
num_tokens = tokens.size(1)
tokens_per_second = num_tokens / elapsed_time

print("Generated Text:", generated_text)
print("Tokens per second:", tokens_per_second)

