from transformers import pipeline
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

messages = [
    [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ],
    [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ],
    [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ],
    [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ],
    [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ],
    [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ],
    [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ],
    [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ],
    [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ],
    [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ],
    [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ],
    [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ],
    [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ],
    [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ],
    [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ],
    [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ],
    [
        {"role": "system", "content": "You are a friendly AI assistant."},
        {"role": "user", "content": "What's the weather like?"},
    ]
]

# pipelineの作成
generator = pipeline(
        'text-generation',
        model='meta-llama/Meta-Llama-3.1-8B-Instruct',
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
        )
print("pad token was set to {}".format(generator.tokenizer.pad_token_id))
print()
print("eos token is {}".format(generator.model.config.eos_token_id))

generator.tokenizer.pad_token_id = generator.model.config.eos_token_id[0]

print("pad token is now set to {}".format(generator.tokenizer.pad_token_id))
print("tokenizer.eos_token_id is list? => {}".format(tokenizer.eos_token_id))

# バッチ処理の実行
# outputs = generator(messages, batch_size=17, max_new_tokens=50, do_sample=False,pad_token_id=tokenizer.eos_token_id)
outputs = generator(messages, batch_size=17, max_new_tokens=50, do_sample=False,pad_token_id=generator.model.config.eos_token_id[0])


print("-------------------------")
for output in outputs:
    print(output[0]['generated_text'])
    print("------------------------")
    
