import torch
import transformers

# text-generationパイプラインの初期化
#generator = pipeline("text-generation", model="gpt2")
model_id = "meta-llama/Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    batch_size=2
)

pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id[0]

# バッチ処理の例とバッチサイズの指定
texts = ["おはようございます", "今日はいい天気ですね"]
outputs = pipeline(texts, max_length=50, batch_size=2)  # バッチサイズを2に指定

# 出力の確認
for i, output in enumerate(outputs):
    print(f"入力 {i+1}: {texts[i]}")
    print(f"出力 {i+1}: {output[0]['generated_text']}\n")

