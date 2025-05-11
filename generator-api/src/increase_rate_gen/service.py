from .schema import IncreaseRateGenRequestDto

from .method.generator import increase_rate_generator
from .method.model_loader import load_model


def increase_rate_gen_service(request: IncreaseRateGenRequestDto):
    prompt = ""
    tokenizer, model = load_model()

    result = increase_rate_generator(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=280
    )
    return "Testing"
