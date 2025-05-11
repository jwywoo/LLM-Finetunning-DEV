from pydantic import BaseModel


class RmWeight(BaseModel):
    bench_press: int
    squat: int
    dead_lift: int
    over_head_press: int


class IncreaseRateGenRequestDto(BaseModel):
    experience: int
    gender: str
    purpose: int
    rm_weight: RmWeight


class IncreaseRateGenResponseDto(BaseModel):
    program: str
    init_weight_rate: int
    increase_rate_week: int
    increase_rate_set: int
    deloading_rate: int
    weekly_weight_increase_plan: str
