from fastapi import FastAPI, APIRouter, Depends, UploadFile, status
from fastapi.responses import JSONResponse
import os
from helpers.config import Settings, get_settings
from controllers import DataController, ProjectController, ProcessController
from models import ResponseSignal
import aiofiles
import logging
from .schemes.data import ProcessReuest
logger = logging.getLogger(__name__)


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
    
    file_path, file_id = data_controller.generate_unique_filepath(
        orig_file_name=file.filename,
        project_id=project_id
    )

    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:

        logger.error(f"Error while uploading file: {e}")

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.FILE_UPLOAD_FAILED.value
            }
        )

    return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "signal": ResponseSignal.File_Uploaded_Successfully.value,
                "file_id": file_id
            }
        )

@data_router.post("/process/{project_id}")
async def process_endpoint(
    project_id: str,
    process_request: ProcessReuest,
):
    file_id = process_request.file_id
    chunk_size = process_request.chunk_size
    overlap_size = process_request.overlap_size
    process_controller = ProcessController(project_id=project_id)
    try:
        file_content = process_controller.get_file_content(file_id=file_id)
    except FileNotFoundError as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": str(e)}
        )
    except ValueError as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": str(e)}
        )
    if len(file_content) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.Processing_Failed.value}
        )
    file_chunks = process_controller.process_file_content(
        file_content=file_content,
        file_id=file_id,
        chunk_size=chunk_size,
        overlap_size=overlap_size
    )
    if file_chunks is None or len(file_chunks) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.Processing_Failed.value}
        )
    return file_chunks

