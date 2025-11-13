from vllm import LLM, SamplingParams

import time

if __name__ == "__main__":
    model_path = "path/to/Marco-MT-Algharb"
    llm = LLM(model=model_path)

    with open("wmttest2022.zh", "r") as file:
        dataset = file.read().split('\n')

    # Ignore last line
    dataset = dataset[:-1]

    prompts_to_generate = []

    for line in dataset:
        prompt = f"Human: Please translate the following text into english: \n{line}<|im_end|>\nAssistant:"
        prompts_to_generate += [prompt]

    sampling_params = SamplingParams(n=1, temperature=0.001, top_p=0.001, max_tokens=512)

    outputs = llm.generate(prompts_to_generate, sampling_params)

    buffer = ""

    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        buffer += generated_text + "\n"
    
    with open("Marco-MT.en", "w") as file:
        file.write(buffer)
