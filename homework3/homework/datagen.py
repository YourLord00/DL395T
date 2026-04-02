def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    #load cot model and data
    from .cot import CoTModel
    from .data import Dataset, is_answer_valid

    model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    dataset = Dataset("train")

    questions = [dataset[i][0] for i in range(len(dataset))]
    prompts = [model.format_prompt(q) for q in questions]

    batched_answers = model.batched_generate(prompts, num_return_sequences=oversample, temperature=temperature)

    rft_data = []
    # iterate datasets
    for i in range(len(dataset)):
        # get standard answers and response from data/train.json
        question, the_answer = dataset[i]
        # no early break 
        for response in batched_answers[i]:
            predicted = model.parse_answer(response)
            if is_answer_valid(predicted, the_answer):
                rft_data.append([question, the_answer, response])
                # break

    import json
    with open(output_json, "w") as f:
        json.dump(rft_data, f, indent=2)

    print(f"Generated {len(rft_data)} / {len(dataset)} samples")

if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
