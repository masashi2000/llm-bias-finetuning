import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# pad_token_idを設定
tokenizer.pad_token_id = tokenizer.eos_token_id

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

print("aaaaaaaaaaaaaaaaaaaaaaaaaa")

# 会話形式のバッチ処理
messages = [
    [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ],
    [
        {"role": "system", "content": "You are a friendly AI assistant."},
        {"role": "user", "content": "What's the weather like?"},
    ]
]
pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id
outputs = pipeline(
    messages,
    max_new_tokens=512,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=False,
    batch_size=2
)

print(outputs)

