from .schema import IncreaseRateGenRequestDto, IncreaseRateGenResponseDto

from .method.generator import increase_rate_generator
from .method.model_loader import load_model
from .method.formatter import prompt_formatter, response_parser


def increase_rate_gen_service(request: IncreaseRateGenRequestDto):
    formatted_prompt = prompt_formatter(
        request=request
    )
    # Model & Tokenizer loading
    tokenizer, model = load_model()

    generated_response = increase_rate_generator(
        model=model,
        tokenizer=tokenizer,
        prompt=formatted_prompt,
        max_new_tokens=280
    )
    parsed_response = response_parser(generated_response)
    return IncreaseRateGenResponseDto(
        program=parsed_response['program'],
        init_weight_rate=int(parsed_response['init_weight_rate']),
        increase_rate_week=int(parsed_response['increase_rate_week']),
        increase_rate_set=int(parsed_response['increase_rate_set']),
        deloading_rate=int(parsed_response['deloading_rate']),
        weekly_weight_increase_plan=parsed_response['weekly_weight_increase_plan']
    )
