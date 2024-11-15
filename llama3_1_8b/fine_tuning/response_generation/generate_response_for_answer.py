import transformers
import torch
import yaml
import pandas as pd
import argparse
import csv
import os
import time
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='Path to the model config file')
    parser.add_argument('--question_and_answer_file', type=str, required=True, help='Path to the answer file')
    parser.add_argument('--persona_file', type=str, required=True, help='Path to the persona file')
    parser.add_argument('--template_file', type=str, required=True, help='Path to the template file')
    parser.add_argument('--name_file', type=str, required=True, help='Path to name file')


    args = parser.parse_args()

    # Load config file
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)
    print("\nModel is below:")
    print(config["model_config"]["model_id"])

    # Initialize model
    model_id = config["model_config"]["model_id"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = transformers.pipeline(
        "text-generation",
        model=model_id,
        device=device,
        torch_dtype=torch.bfloat16
    )

    # Handle padding configuration for different models
    if model_id == "meta-llama/Llama-3.1-8B-Instruct":
        generator.tokenizer.pad_token_id = generator.model.config.eos_token_id[0]
        generator.tokenizer.padding_side = 'left'
    elif model_id == "mistralai/Mistral-7B-Instruct-v0.3":
        generator.tokenizer.pad_token_id = generator.model.config.eos_token_id
    else:
        raise ValueError("Unknown model configuration. Please add the model configuration.")

    # Load questions and answers
    questions_and_answers_df = pd.read_csv(args.question_and_answer_file)

    # Load persona
    with open(args.persona_file, "r", encoding="utf-8") as f:
        persona = f.read().strip()
    print("\nPersona is below:")
    print(persona)

    # Load template file
    with open(args.template_file, "r", encoding="utf-8") as f:
        template = f.read().strip()
    print("\nTemplate is below:")
    print(template)

    # Load names
    names = pd.read_csv(args.name_file)
    names = names["Name"].tolist()

    # Results DataFrame for saving
    results_df = pd.DataFrame(columns=questions_and_answers_df.columns)
    results_df["Response"] = ""

    # Prepare output file name
    question_and_answer_file_name = os.path.splitext(os.path.basename(args.question_and_answer_file))[0]
    persona_file_name = os.path.splitext(os.path.basename(args.persona_file))[0]
    template_file_name = os.path.splitext(os.path.basename(args.template_file))[0]

    # Check whether agreement or disagreement
    if ("dem" in question_and_answer_file_name) and ("dem" in persona_file_name):
        output_file = f"{question_and_answer_file_name}_{persona_file_name}_{template_file_name}.csv"

    elif ("dem" in question_and_answer_file_name) and ("rep" in persona_file_name):
        output_file = f"{question_and_answer_file_name}_{persona_file_name}_{template_file_name}.csv"

    elif ("rep" in question_and_answer_file_name) and ("rep" in persona_file_name):
        output_file = f"{question_and_answer_file_name}_{persona_file_name}_{template_file_name}.csv"

    elif ("rep" in question_and_answer_file_name) and ("dem" in persona_file_name):
        output_file = f"{question_and_answer_file_name}_{persona_file_name}_{template_file_name}.csv"
    
    # Create prompts substituting persona, question and answer with template.
    prompts = [
        [
            {"role": "system", "content": persona},
            {"role": "user", "content": template.format(
                name=random.choice(names),
                question=row["Rephrased Question"],
                answer=row["Answer"]
                )
            }
        ]
        for _, row in questions_and_answers_df.iterrows() 
    ]
        
    # Generate response with batch processing
    # バッチ処理していく llama3.1-8bならXぐらいが限界に見えた
    outputs = generator(
        prompts,
        batch_size=2,
        do_sample=True,
        max_new_tokens=1000,
        top_p=1,
        temperature=1.0,
        pad_token_id=generator.tokenizer.pad_token_id,
    )
    # Process the outputs and store the generated text
    generated_responses = [
        output[0]["generated_text"][-1]["content"] if isinstance(output, list) else output
        for output in outputs
    ]
    # Add the responses to the DataFrame under the "Response" column
    questions_and_answers_df["Response"] = generated_responses
    
    # Save results to csv
    questions_and_answers_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"Responses saved to {output_file}")

if __name__ == "__main__":
    start_time = time.time()

    main()

    end_time = time.time()
    elapsed_time = end_time - start_time

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)

    print(f"\nDone in {hours} hours {minutes} minutes!!\n")
