from controllers.BaseController import BaseController
from fastapi import UploadFile

class DataController(BaseController):
    def __init__(self):
        super().__init__()

    def validate_upload_file(self, file: UploadFile):
        if file.content_type not in self.app_settings.FILE_ALLOWED_TYPES:
            return False, "No file provided."
        if file.size > self.app_settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            return False, "File size exceeds the maximum limit."
        return True, "File is valid."