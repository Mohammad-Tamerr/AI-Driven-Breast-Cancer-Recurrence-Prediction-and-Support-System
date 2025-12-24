from helpers.config import Settings, get_settings
import os

class BaseController:
    def __init__(self):
        self.app_settings = get_settings()  # Load settings once for the controller

    