import uvicorn

from app.core.settings import settings


def run():
    """
    Run FastAPI application using uvicorn.
    """
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENV == "development"
    )


if __name__ == "__main__":
    run()