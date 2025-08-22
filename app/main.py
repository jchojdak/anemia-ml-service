from fastapi import FastAPI
from app.api.v1.detect_controller import router as detect_router
from app.core.config import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug
    )

    app.include_router(detect_router, prefix="/api/v1")

    return app

app = create_app()