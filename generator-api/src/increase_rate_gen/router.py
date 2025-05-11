from fastapi import APIRouter

from .schema import IncreaseRateGenRequestDto, IncreaseRateGenResponseDto
from .service import increase_rate_gen_service

router = APIRouter()


@router.post("/gen/routine/rate")
def increase_rate_gen_router(request: IncreaseRateGenRequestDto):
    return increase_rate_gen_service(request)
