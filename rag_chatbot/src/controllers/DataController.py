from controllers.BaseController import BaseController
from fastapi import UploadFile
from models import ResponseSignal
import os

class DataController(BaseController):
    def __init__(self):
        super().__init__()
        self.size_scale = 1048576 # convert MB to bytes
    def validate_upload_file(self, file: UploadFile):
        if file.content_type not in self.app_settings.FILE_ALLOWED_TYPES:
            return False, ResponseSignal.File_Type_Error.value
        
        if file.size > self.app_settings.MAX_FILE_SIZE_MB * self.size_scale:
            return False, ResponseSignal.File_Size_Error.value
        
        return True, ResponseSignal.File_Valid.value,ResponseSignal.File_Uploaded_Successfully.value