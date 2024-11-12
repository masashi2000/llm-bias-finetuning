from transformers import pipeline
import torch

# pipelineの作成
generator = pipeline(
        'text-generation',
        model='meta-llama/Llama-3.1-8B-Instruct',
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
        )
print("pad token was set to {}".format(generator.tokenizer.pad_token_id))
print()
print("eos token is {}".format(generator.model.config.eos_token_id))

generator.tokenizer.pad_token_id = generator.model.config.eos_token_id[0]

print("pad token is now set to {}".format(generator.tokenizer.pad_token_id))

# 入力テキストのリスト
texts = [
    "Once upon a time",
    "In a galaxy far far away", 
    "It was a dark and stormy night"
]

# バッチ処理の実行
outputs = generator(texts, batch_size=3, max_length=50)

print("-------------------------")
for output in outputs:
    print(output[0]['generated_text'])
    print("------------------------")
    
