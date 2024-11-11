import transformers
import torch
import sys
import csv
import os

if len(sys.argv) != 2:
    print("Usage: python script.py <prompt_file>")
    sys.exit(1)

prompt_file = sys.argv[1]

# Read the prompt from the specified file
with open(prompt_file, 'r', encoding='utf-8') as f:
    prompt_text = f.read()


# Initialize the pipeline with the specified settings
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
avairable_device = "cuda" if torch.cuda.is_available() else "cpu"

generator = transformers.pipeline(
    model=model_id,
    device=avairable_device,
    torch_dtype=torch.bfloat16
)


# Generate 50 personas
personas = []

prompt = [{"role": "system", "content": "You are a helpful assistant."}]
prompt.append({"role": "user", "content": prompt_text})

for i in range(100):
    generation = generator(
        prompt,
        temperature=1.0,
        top_p=1,
        max_new_tokens=512,
        do_sample=True
    )
    personas.append(generation[0]["generated_text"][2]["content"])
    print("{} : {}".format(i+1, generation[0]["generated_text"][2]["content"]))
    print()

# Prepare the CSV file name based on the prompt file name
base_name = os.path.splitext(prompt_file)[0]
csv_file = base_name + '.csv'

# Write the generated personas to a CSV file
with open(csv_file, 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'Persona'])
    for i, persona in enumerate(personas, 1):
        writer.writerow([i, persona])

