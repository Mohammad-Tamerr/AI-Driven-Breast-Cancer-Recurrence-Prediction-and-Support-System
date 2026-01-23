from fastapi import APIRouter, Depends
import os
from helpers.config import get_settings, Settings


base_router = APIRouter(
    prefix="/Rafeek/v1",
    tags=["Rafeek_v1"],
)

@base_router.get("/")
async def welcome(app_settings: Settings = Depends(get_settings)):
    return {
        "app_name": app_settings.APP_NAME,
        "app_version": app_settings.APP_VERSION,
    }
