from fastapi import FastAPI
from app.api.routes import router
from app.core.logging_config import configure_logging
from app.core.firebase_config import initialize_firebase

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    initialize_firebase()

app.include_router(router)
configure_logging()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")