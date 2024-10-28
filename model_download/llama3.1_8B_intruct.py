"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデルとトークナイザーの読み込み
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# GPUが利用可能か確認し、利用可能ならモデルをGPUに移動
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 使用するデバイスを表示
print(f"使用するデバイス: {device}")

# 無限ループで入力を受け取る
while True:
    # ユーザーからの入力を受け取る
    user_input = input("ユーザー: ")

    # 終了コマンド
    if user_input.lower() == "終了":
        print("プログラムを終了します。")
        break

    # モデルで応答を生成
    inputs = tokenizer(user_input, return_tensors="pt").to(device)  # 入力もデバイスに移動

    # 生成時間の計測開始
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=50, temperature=1.0)
    end_time = time.time()

    # 応答をデコード
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # トークン生成速度の計算と表示
    num_tokens = outputs.shape[-1]  # 生成されたトークン数
    time_taken = end_time - start_time
    tokens_per_second = num_tokens / time_taken
    print("モデル: " + response)
    print(f"生成速度: {tokens_per_second:.2f} トークン/秒")
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデルとトークナイザーの読み込み
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# GPUが利用可能か確認し、利用可能ならモデルをGPUに移動
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 使用するデバイスを表示
print(f"使用するデバイス: {device}")

# 無限ループで入力を受け取る
while True:
    # ユーザーからの入力を受け取る
    user_input = input("ユーザー: ")

    # 終了コマンド
    if user_input.lower() == "終了":
        print("プログラムを終了します。")
        break

    # モデルで応答を生成
    inputs = tokenizer(user_input, return_tensors="pt").to(device)  # 入力もデバイスに移動

    # 生成時間の計測開始
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=1000, temperature=1.0)
    end_time = time.time()

    # 応答をデコード
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # トークン生成速度の計算と表示
    num_tokens = outputs.shape[-1]  # 生成されたトークン数
    time_taken = end_time - start_time
    tokens_per_second = num_tokens / time_taken
    print("モデル: " + response)
    print(f"生成速度: {tokens_per_second:.2f} トークン/秒")

