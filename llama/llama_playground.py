
from transformers import AutoTokenizer, pipeline
import torch
from huggingface_hub import login
login("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")



def play():
    model = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipe = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    pipe("The key to life is")







if __name__ == "__main__":
    play()
