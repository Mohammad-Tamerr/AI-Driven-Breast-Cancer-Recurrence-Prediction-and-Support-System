from pydantic import BaseModel
from typing import Optional as optional

class ProcessReuest(BaseModel):
    file_id: str
    chunk_size: optional[int] = 100
    overlap_size: optional[int] = 20
    do_reset : optional[int] = 0 
