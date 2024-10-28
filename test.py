import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "Be a helpful assistant."},
    {"role": "user", "content": "Create a persona of a passionate Democrat partisan with the following ideology: Believes climate change is a big problem. Believes racism is a big problem. Believes gun violence is a big problem. Doesn't believe that illegal immigration is a big problem. Use the second person singular. Do not assign a name to it. Please answer in 100 words or fewer."},
]

outputs = pipeline(
    messages,
    max_new_tokens=512,
)
print(outputs[0]["generated_text"][-1])
