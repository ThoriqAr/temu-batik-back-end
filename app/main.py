from fastapi import FastAPI
from app.api.router import api_router

app = FastAPI(
    title="Temu Batik API",
    description="Backend API for Batik Classification and Explainable AI",
    version="1.0.0"
)

app.include_router(api_router)