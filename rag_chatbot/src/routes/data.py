from fastapi import FastAPI, APIRouter, Depends, UploadFile, status
from fastapi.responses import JSONResponse
import os
from helpers.config import Settings, get_settings
from controllers import DataController
from models import ResponseSignal
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

    data_controller = DataController()

    is_valid, result_signal = data_controller.validate_upload_file(file=file)
    if not is_valid:
        return JSONResponse(
            status_code=400,
            content={
                "signal": result_signal
            }
        )
    else: 
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "signal": ResponseSignal.SUCCESS.value,
                "message": ResponseSignal.SUCCESS.value,
                "project_id": project_id,
                "filename": file.filename
            }
        )