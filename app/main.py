from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.settings import settings
from app.core.logger import setup_api_logger, get_logger

# -------------------------
# Setup Logging
# -------------------------
setup_api_logger()
logger = get_logger()

logger.info("🚀 Starting API...")

# -------------------------
# Initialize App
# -------------------------
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Heart Failure Prediction API with SHAP Explainability"
)

# -------------------------
# CORS (important for frontend later)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Include Routes
# -------------------------
app.include_router(api_router, prefix="/api/v1")

# -------------------------
# Root Endpoint
# -------------------------
@app.get("/")
def root():
    return {
        "message": "Heart Failure ML API",
        "version": settings.VERSION,
        "docs": "/docs"
    }


# -------------------------
# Startup Event
# -------------------------
@app.on_event("startup")
def startup_event():
    logger.info("✅ API startup complete")


# -------------------------
# Shutdown Event
# -------------------------
@app.on_event("shutdown")
def shutdown_event():
    logger.info("🛑 API shutting down")