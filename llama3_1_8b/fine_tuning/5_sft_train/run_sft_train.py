from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
import torch
import os
import wandb
import random
import argparse
import yaml


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_file', type=str, required=True, help='Path to datset file')
parser.add_argument('--config_file', type=str, required=True, help='Path to config file')
args = parser.parse_args()


with open(args.config_file, 'r') as f:
    config = yaml.safe_load(f)

# Try small batch try to increase number of steps.
# Memory wise can be increased up to 32 wihtout any issues.
BATCH_SIZE = config['model_config']['sft_batch_size']

# make dataset name
dataset_name = os.path.splitext(os.path.basename(args.dataset_file))[0]

class EvaluateFirstStepCallback(TrainerCallback):
    """ Callback which makes the trainer evaluate at the first step. """
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True


def run_training(run_name, model, tokenizer, lora_r, lora_a, epochs, rnd_seed, attempt):
    peft_config = LoraConfig(
        lora_alpha=lora_a,
        r=lora_r,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM", 
        target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj",'up_proj','down_proj'],
    )

    # D driveに保存する
    # config['model_config']['sft_model_base_dir'] == "/mnt/d/models/{MODEL_NAME}/sft"
    output_dir = config['model_config']['sft_model_base_dir']

    training_arguments = TrainingArguments(
        output_dir= output_dir + f"/{dataset_name}/attempt{attempt}/" + run_name,
        per_device_train_batch_size=BATCH_SIZE,
        #per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        save_steps=200,
        logging_steps=20,
        learning_rate=2e-4,
        fp16=True,
        #evaluation_strategy="steps",
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_ratio = 0.1,
        run_name=run_name,
        #report_to='wandb',
        logging_dir='.logs',
        seed=rnd_seed,
    )

    #_ = wandb.init(project= dataset_name, name=run_name)

    train_dataset = load_dataset("json", data_files=args.dataset_file).shuffle(seed=rnd_seed)["train"]
    
    #test_dataset = load_dataset("csv", data_files="datasets/republican_2K/test/dataset.csv").shuffle(seed=rnd_seed)["train"]

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        #eval_dataset=test_dataset, # remove you have low VRAM and getting OOM errors
        peft_config=peft_config,
        #dataset_text_field="response",
        max_seq_length=config["model_config"]["sft_max_seq_length"],
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )
    #trainer.add_callback(EvaluateFirstStepCallback())

    trainer.train()

MODEL_NAME = config["model_config"]["model_id"]

# ここから修正
RUN_NAME = dataset_name

# Load model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=getattr(torch, "float16"),
    bnb_4bit_use_double_quant=False,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto",
)
# いったんここはなしで行こう
# tokenizer.padding_side = "right"
model.config.pad_token_id = tokenizer.pad_token_id = tokenizer.eos_token_id

attempts = 1
# rs = [16, 64, 128, 256]
rs = [128]
for i in range(attempts):
    for r in rs:
        alpha = r * 2
        run_name = f"{RUN_NAME}_r{r}_a{alpha}_b{BATCH_SIZE}"
        run_training(run_name=run_name, model=model, tokenizer=tokenizer, lora_r=r,
                     lora_a=alpha, epochs=1, rnd_seed=random.randint(1, 100000), attempt=i + 1)
