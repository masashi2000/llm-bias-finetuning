import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデルとトークナイザーをロード
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model = model.to("cuda")  # GPUを使用

# 入力テキスト
input_text = "What are the advantages of using LLMs in research?"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# 生成プロセスの時間計測を開始
start_time = time.time()

# テキスト生成
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=100, do_sample=True, top_p=0.9, temperature=1.0)

# 結果をデコード
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Text:\n", generated_text)

# トークン数と実行時間の取得
end_time = time.time()
total_time = end_time - start_time
num_tokens = outputs.shape[1]
tokens_per_second = num_tokens / total_time

print(f"\nTotal Time: {total_time:.2f} seconds")
print(f"Number of Tokens: {num_tokens}")
print(f"Tokens per Second: {tokens_per_second:.2f} tokens/second")

