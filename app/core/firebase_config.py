import os
import base64
import json
from firebase_admin import credentials, initialize_app, storage
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from the .env file

def initialize_firebase():
    # Decode the base64 encoded Firebase key from the environment variable
    firebase_key_base64 = os.getenv('FIREBASE_KEY_BASE64')
    if not firebase_key_base64:
        print("Environment variable FIREBASE_KEY_BASE64 is not set or is empty")
        return None

    try:
        # Decode the base64 string to get the JSON string
        firebase_key_json = base64.b64decode(firebase_key_base64).decode('utf-8')

        # Convert the JSON string into a Python dictionary
        firebase_key_dict = json.loads(firebase_key_json)

        # Create credentials from the dictionary instead of a file
        cred = credentials.Certificate(firebase_key_dict)

        # Initialize the Firebase app with the credentials and storage bucket
        initialize_app(cred, {
            'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET')  # Ensure this environment variable is set
        })

        # Get a reference to the storage bucket
        return storage.bucket()

    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        return None

firebase_bucket = initialize_firebase()
