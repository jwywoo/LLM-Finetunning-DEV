import json
import re

from ..schema import IncreaseRateGenRequestDto


def prompt_formatter(request: IncreaseRateGenRequestDto):
    input_text = [
        "## 질문",
        "-5/3/1 프로그램",
        f"- **운동경험**: {request.experience}",
        f"- **성별**: {request.gender}",
        f"- **운동목표**: {request.purpose}",
        "- **1RM**:",
        f"  - 벤치프레스: {request.rm_weight.bench_press}",
        f"  - 스쿼트: {request.rm_weight.squat}",
        f"  - 데드리프트: {request.rm_weight.dead_lift}",
        f"  - 오버헤드프레스: {request.rm_weight.over_head_press}"
    ]
    return "\n".join(input_text)+"\n\n## 답변\n"


def response_parser(generated_response: str) -> dict:
    response_text = generated_response.split("<END>")[0]

    match = re.search(r"## 답변\s*(\{.*\})", response_text, re.DOTALL)
    if not match:
        raise ValueError("Gen failed :(")

    json_block = match.group(1).strip()
    print(json_block)

    try:
        parsed = json.loads(json_block)
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"parsing failed :(")
