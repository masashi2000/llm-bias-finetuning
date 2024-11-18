from transformers import AutoTokenizer

# Load the Llama 3 tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Retrieve the list of all special tokens
special_tokens = tokenizer.all_special_tokens

# Check if '<|finetune_right_pad_id|>' is among the special tokens
if '<|finetune_right_pad_id|>' in special_tokens:
    print("The '<|finetune_right_pad_id|>' token is present in the tokenizer.")
else:
    print("The '<|finetune_right_pad_id|>' token is NOT present in the tokenizer.")

print(special_tokens)

