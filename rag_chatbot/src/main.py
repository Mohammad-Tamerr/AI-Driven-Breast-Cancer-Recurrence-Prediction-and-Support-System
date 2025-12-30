from fastapi import FastAPI
from routes import base, data
# from helpers.config import get_settings
# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from stores.LLM.LLMProviderFactory import LLMProviderFactory 

app = FastAPI()

app.include_router(base.base_router)
app.include_router(data.data_router)