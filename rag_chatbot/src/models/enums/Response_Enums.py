from enum import Enum

class ResponseSignal(Enum):
    File_Valid = "File is valid."
    File_Type_Error = "File type is not allowed."
    File_Uploaded_Successfully = "Uploaded successfully."
    File_Size_Error = "File size exceeds the maximum limit."
    FILE_UPLOAD_FAILED = "File upload failed."