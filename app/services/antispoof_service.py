import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json
from pathlib import Path
import logging
from app.core.config import MODEL_PATH, CONFIG_PATH
from app.utils.file_utils import download_file, remove_temp_file
from fastapi import HTTPException
from app.core.firebase_config import firebase_bucket
from io import BytesIO
from PIL import Image

SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']

class AntispoofService:
    def __init__(self):
        self.model = None
        self.config = None
        self.load_model_and_config()

    def load_model_and_config(self):
        try:
            if not Path(MODEL_PATH).exists():
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            self.model = load_model(MODEL_PATH)
            logging.info(f"Model loaded successfully from {MODEL_PATH}")

            if not Path(CONFIG_PATH).exists():
                raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
            with open(CONFIG_PATH, "r") as config_file:
                self.config = json.load(config_file)
            logging.info(f"Configuration loaded successfully from {CONFIG_PATH}")
        except Exception as e:
            logging.error(f"Error loading model or config: {str(e)}")
            raise

    def verify_image_format(self, filename: str):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in SUPPORTED_IMAGE_FORMATS:
            raise ValueError(f"Unsupported image format. Supported formats are: {', '.join(SUPPORTED_IMAGE_FORMATS)}")

    async def verify_image(self, image_url: str):
        if self.model is None or self.config is None:
            raise ValueError("Model or configuration not loaded")

        firebase_filename = None
        try:
            firebase_filename = await download_file(image_url)

            # Verify image format
            self.verify_image_format(firebase_filename)

            # Download the file from Firebase
            blob = firebase_bucket.blob(firebase_filename)
            image_bytes = blob.download_as_bytes()
            
            # Process the image
            image = Image.open(BytesIO(image_bytes))
            image = image.resize((self.config["img_width"], self.config["img_height"]))
            image_array = img_to_array(image)
            image_array = image_array / 255.0  # Normalize
            image_array = np.expand_dims(image_array, axis=0)

            prediction = self.model.predict(image_array)[0][0]

            is_real = bool(prediction <= self.config["threshold"])
            result = "Real" if is_real else "Spoof"

            return {
                "image_url": image_url,
                "prediction_score": float(prediction),
                "is_real": is_real,
                "result": result
            }
        except ValueError as e:
            logging.error(f"Invalid image format: {str(e)}")
            return {"error": str(e), "detail": "Unsupported image format"}
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
        finally:
            if firebase_filename:
                await remove_temp_file(firebase_filename)

antispoof_service = AntispoofService()