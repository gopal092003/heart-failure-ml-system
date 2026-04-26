from fastapi import APIRouter
from app.api.v1.endpoints import predict, explain, health

api_router = APIRouter()

api_router.include_router(predict.router, prefix="/predict", tags=["Predict"])
api_router.include_router(explain.router, prefix="/explain", tags=["Explain"])
api_router.include_router(health.router, prefix="/health", tags=["Health"])