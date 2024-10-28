
from transformers import AutoTokenizer, pipeline
import torch
from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()
login(os.getenv('HUGGINF_FACE'))


def play():
    model = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipe = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    test_output = pipe("The key to life is")
    print(test_output)



if __name__ == "__main__":
    play()
