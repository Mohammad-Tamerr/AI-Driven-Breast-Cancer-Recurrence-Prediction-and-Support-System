from fastapi import APIRouter, Depends, UploadFile
import os
from helpers.config import Settings, get_settings
from controllers import DataController

data_router = APIRouter(
    prefix="/Rafeek/v1/data",
    tags=["Rafeek_v1, Data"],
)

@data_router.post("/upload/{project_id}")
async def upload_file(
    project_id: str,
    file: UploadFile,
    app_settings: Settings = Depends(get_settings)
):
    
    is_valid = DataController().validate_upload_file(file)
    return is_valid
