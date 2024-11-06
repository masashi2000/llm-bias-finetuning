import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = 'meta-llama/Llama-3.1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

input1 = [{"role": "user", "content": "Hi, how are you?"}]
input2 = [{"role": "user", "content": "How are you feeling today?"}]
texts = tokenizer.apply_chat_template([input1, input2], add_generation_prompt=True, tokenize=False)

tokenizer.pad_token_id = tokenizer.eos_token_id  # Set a padding token
inputs = tokenizer(texts, padding="longest", return_tensors="pt")
inputs = {key: val.to(model.device) for key, val in inputs.items()}

generation = model.generate(**inputs, max_new_tokens=512)
print(generation)

