from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.router import api_router
from app.core.model_registry import load_models
from app.core.embedding_extractor_registry import load_embedding_extractors
from app.core.embedding_database_registry import load_embedding_databases


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting Temu Batik API...")

    print("Loading models...")
    load_models()

    print("Loading embedding extractors...")
    load_embedding_extractors()

    print("Loading embedding databases...")
    load_embedding_databases()

    print("Startup completed.")

    yield

    print("Shutting down application...")


app = FastAPI(
    title="Temu Batik API",
    description="Backend API for Batik Classification and Explainable AI",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_router)