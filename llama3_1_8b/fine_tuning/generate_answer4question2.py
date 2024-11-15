import transformers
import torch
import yaml
import pandas as pd
import argparse
import csv
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--persona_file', type=str, required=True, help='Path to the persona file')
    parser.add_argument('--question_file', type=str, required=True, help='Path to the question file')
    parser.add_argument('--config_file', type=str, required=True, help='Path to the model config file')
    args = parser.parse_args()

    # Load config file
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

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

    # Load persona
    with open(args.persona_file, "r", encoding="utf-8") as f:
        persona = f.read().strip()
    print("Persona is below:")
    print(persona)

    # Load questions
    questions_df = pd.read_csv(args.question_file)
    questions = questions_df["Rephrased Question"].tolist()

    # Prepare output file name
    persona_file_name = os.path.splitext(os.path.basename(args.persona_file))[0]
    question_file_name = os.path.splitext(os.path.basename(args.question_file))[0]
    output_file = f"{persona_file_name}_answers_for_{question_file_name}.csv"

    # Add a new column for responses
    responses = []

    # Generate responses for each question
    for question in questions:
        question_responses = []
        print("--------------------------")
        print("\nQuestion is below:")
        print(question)
        for _ in range(20):  # Run 20 times per question
            prompt = [
                {"role": "system", "content": persona},
                {"role": "user", "content": question}
            ]
            output = generator(
                prompt,
                do_sample=True,
                max_new_tokens=600,
                top_p=1,
                temperature=1.0,
                pad_token_id=generator.tokenizer.pad_token_id,
            )
            # Extract the response
            generated_text = output[0]["generated_text"][-1]["content"] if isinstance(output[0], dict) else output[0]
            print()
            print(generated_text)
            question_responses.append(generated_text)
        
        responses.append(question_responses)

    # Write to a new CSV file with responses
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["Original Question", "Rephrased Question", "Responses"])
        # Write each question and its 20 responses
        for original_question, rephrased_question, response_list in zip(
                questions_df["Original Question"], questions_df["Rephrased Question"], responses):
            writer.writerow([original_question, rephrased_question, "; ".join(response_list)])

    print(f"Responses saved to {output_file}")

if __name__ == "__main__":
    main()

