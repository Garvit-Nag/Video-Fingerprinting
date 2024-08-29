import os

TEMP_STORAGE = os.path.join(os.getcwd(), "temp_storage")
os.makedirs(TEMP_STORAGE, exist_ok=True)