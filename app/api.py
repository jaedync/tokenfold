"""GET /api/stats - returns dashboard JSON blob."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .aggregator import build_dashboard_data, get_cache_version

router = APIRouter()


@router.get("/api/stats/version")
async def stats_version():
    return {"version": get_cache_version()}


@router.get("/api/stats")
async def stats():
    data = build_dashboard_data()
    return JSONResponse(content=data)
