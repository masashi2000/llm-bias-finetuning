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

    # Results culumns
    results_df = pd.DataFrame(columns=questions_df.columns)
    results_df["Response"] = ""

    # Prepare output file name
    persona_file_name = os.path.splitext(os.path.basename(args.persona_file))[0]
    question_file_name = os.path.splitext(os.path.basename(args.question_file))[0]
    output_file = f"{question_file_name}_{persona_file_name}.csv"

    # Add a new column for responses
    """
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
                print("-----------------------------")
                print()
                print(generated_text)
                question_responses.append(generated_text)
            
            responses.append(question_responses)
    """
    
    for index, row in questions_df.iterrows():
        
        question_responses = []
        print("-------------------------")
        print("\nQuestion is below:")
        print(row["Rephrased Question"])
        # ２０このプロンプトを生成する
        prompts = [
            [
                {"role": "system", "content": persona},
                {"role": "user", "content": row["Rephrased Question"]}
            ]
            for _ in range(20)  # Generate 20 identical prompts,
        ]
        # バッチ処理していく llama3.1-8bなら20ぐらいが限界に見えた
        outputs = generator(
            prompts,
            batch_size=20,
            do_sample=True,
            max_new_tokens=1000,
            top_p=1,
            temperature=1.0,
            pad_token_id=generator.tokenizer.pad_token_id,
        )
        # Process the outputs and store the generated text
        for output in outputs:
            # Extract the response content
            generated_text = output[0]["generated_text"][-1]["content"] if isinstance(output, list) else output
            print("-----------------------------")
            print()
            print(generated_text)

            #result_series = pd.Series({"Original Question": row["Original Question"], "Rephrased Question": row["Rephrased Question"], "Response": generated_text})
            #results_df = results_df.append(result_series)
            # 新しい行（Series）を作成
            result_series = pd.Series({"Original Question": row["Original Question"],
                                       "Rephrased Question": row["Rephrased Question"],
                                       "Response": generated_text})

            # DataFrameに新しい行をconcatで追加
            results_df = pd.concat([results_df, result_series.to_frame().T], ignore_index=True)
    
    # Save results to csv
    results_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"Responses saved to {output_file}")

if __name__ == "__main__":
    import time
    start_time = time.time()

    main()

    end_time = time.time()
    elapsed_time = end_time - start_time

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)

    print(f"\nDone in {hours} hours {minutes} minutes!!\n")

