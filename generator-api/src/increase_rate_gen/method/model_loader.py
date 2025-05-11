from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import torch


def load_model():
    model_path = os.path.join(
        "increase_rate_gen/model/task1/", "lora-output-v2")

    print(os.path.exists(model_path))

    base_model_name = "EleutherAI/polyglot-ko-1.3b"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<END>"]})
    tokenizer.eos_token = "<END>"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)

    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, model_path)

    return tokenizer, model
