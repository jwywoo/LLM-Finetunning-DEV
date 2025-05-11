import torch


def increase_rate_generator(prompt, model, tokenizer, max_new_tokens=300, do_sample=False):
    # just to be safe
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # encoding
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    inputs.pop("token_type_ids", None)
    eos_token_id = tokenizer.convert_tokens_to_ids("<END>")

    # generation
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        do_sample=do_sample,
        top_p=0.9 if do_sample else None,
        temperature=0.7 if do_sample else None,
        repetition_penalty=1.1,
    )

    # decoding
    decoded = tokenizer.decode(outputs[0])
    cleaned = decoded.split("<END>")[0].strip() + "\n<END>"
    return cleaned
