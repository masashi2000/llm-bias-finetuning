from transformers import AutoModelForCausalLM

# モデルをロード
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # 適切なモデル名に置き換え
model = AutoModelForCausalLM.from_pretrained(model_name)

# モジュール一覧を取得
for name, module in model.named_modules():
    print(name)

print("-------------------------")
model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # 適切なモデル名に置き換え
model = AutoModelForCausalLM.from_pretrained(model_name)

# モジュール一覧を取得
for name, module in model.named_modules():
    print(name)
