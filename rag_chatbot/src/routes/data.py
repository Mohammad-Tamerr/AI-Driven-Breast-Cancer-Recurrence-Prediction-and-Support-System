from fastapi import FastAPI, APIRouter, Depends, UploadFile, status
from fastapi.responses import JSONResponse
import os
from helpers.config import Settings, get_settings
from controllers import DataController, ProjectController
from models import ResponseSignal
import aiofiles


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
    project_dir_path = ProjectController().get_project_path(project_id=project_id)
    file_path = os.path.join(project_dir_path, file.filename)

    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read(app_settings.CHUNK_SIZE)
        await f.write(content)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "signal": ResponseSignal.File_Uploaded_Successfully.value,
            "message": ResponseSignal.File_Uploaded_Successfully.value,
            "project_id": project_id,
            "filename": file.filename
        }
    )