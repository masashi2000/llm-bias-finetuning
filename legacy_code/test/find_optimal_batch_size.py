from transformers import pipeline
import torch
from transformers import AutoTokenizer
import time
import pynvml

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")


file_path = 'combined_conversation_text.txt'

# ファイルを読み込む
with open(file_path, 'r') as file:
    content = file.read()

# 4人の時
#content = content + "\n" + content

# 3人の時
#with open('combined_conversation_text_half.txt', 'r') as file:
#    content_half = file.read()
#content = content + "\n" + content_half


messages = [
    [
        {"role": "system", "content": "You are passionate Republican. Your name is Michael. You think racism is a big problem."},
        {"role": "user", "content": f"{content}\n\nHi Michael, based on the above conversation, please follow the instruction below:\nThis is the debate about racism. Complete your next reply. Keep your reply shorter than 50 words."},
    ]
]

# pipelineの作成
"""
generator = pipeline(
        'text-generation',
        model='meta-llama/Meta-Llama-3.1-8B-Instruct',
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
        )
"""

available_device = "cuda" if torch.cuda.is_available() else "cpu"
print(available_device)
generator = pipeline(
    "text-generation",
    model='meta-llama/Meta-Llama-3.1-8B-Instruct',
    device=available_device,
    torch_dtype=torch.bfloat16
)
print("pad token was set to {}".format(generator.tokenizer.pad_token_id))
print()
print("eos token is {}".format(generator.model.config.eos_token_id))

generator.tokenizer.pad_token_id = generator.model.config.eos_token_id[0]
generator.tokenizer.padding_side = 'left'

print("pad token is now set to {}".format(generator.tokenizer.pad_token_id))
print("tokenizer.eos_token_id is list? => {}".format(tokenizer.eos_token_id))

# バッチ処理の実行
# outputs = generator(messages, batch_size=17, max_new_tokens=50, do_sample=False,pad_token_id=tokenizer.eos_token_id)

import time

# ログを保存するリスト
time_logs = []

for i in range(5, 10, 1):
    print(f"Start {i}")

    times = []
    for trial in range(3):
        # 処理開始時間を記録
        start_time = time.time()

        # messagesをiの個数分だけ複製
        batched_messages = messages * i  # iの数だけmessagesを繰り返し複製

        # 複製したbatched_messagesをgeneratorに渡す
        outputs = generator(
            batched_messages,
            batch_size=i,
            max_new_tokens=512,
            temperature=1.0,
            pad_token_id=generator.model.config.eos_token_id[0]
        )
        used_memory = torch.cuda.memory_reserved() 
        print(used_memory)
        print()
        print(used_memory/(1024**2))

        # 処理終了時間を記録
        end_time = time.time()

        # 経過時間を計算
        elapsed_time = end_time - start_time
        times.append(elapsed_time)

        # 時間をログに追加
        time_logs.append({
            "batch_size": i,
            "trial": trial + 1,
            "elapsed_time": elapsed_time
        })

        print(f"Trial {trial + 1}: {elapsed_time:.2f} seconds")

    # 平均時間を計算
    average_time = sum(times) / len(times)
    minutes = int(average_time // 60)
    seconds = int(average_time % 60)

    print(f"Average time for batch size {i}: {minutes} minutes {seconds} seconds\n")
    print("\nI'm sleep\n")

# 最終的なログ出力
print("All Time Logs:")
for log in time_logs:
    print(f"Batch Size {log['batch_size']}, Trial {log['trial']}, Time: {log['elapsed_time']:.2f} seconds")

