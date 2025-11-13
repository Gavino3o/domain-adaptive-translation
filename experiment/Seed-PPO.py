from vllm import LLM, SamplingParams

import time

if __name__ == "__main__":
    model_path = "path/to/Seed-X-PPO-7B"
    model = LLM(model=model_path)

    with open("wmttest2022.zh", "r") as file:
        dataset = file.read().split('\n')

    # Ignore last line
    dataset = dataset[:-1]

    prompts_to_generate = []

    for line in dataset:
        prompt = f"Translate the following Chinese sentence into English:\n{line} <en>"
        prompts_to_generate += [prompt]

    decoding_params = SamplingParams(temperature=0, max_tokens=512, skip_special_tokens=True)

    outputs = model.generate(prompts_to_generate, decoding_params)
    
    buffer = ""

    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        buffer += generated_text + "\n"

    with open("Seed-PPO.en", "w") as file:
        file.write(buffer)
