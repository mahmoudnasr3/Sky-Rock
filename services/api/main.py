from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.api.routes.detection import router as detection_router

app = FastAPI(title="Sky-Rock API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection_router)


@app.get("/")
def root():
    return {
        "status": "Sky-Rock backend running",
        "service": "api",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.get("/classes")
def classes():
    return {
        "classes": {
            0: "uav",
            1: "person",
            2: "vehicle",
            3: "two_wheeler",
            4: "airborne_distractor",
        }
    }