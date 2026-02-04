"""
FastAPI Application
Owner: progno-backend agent

Main API entry point for the Progno scenic route planner.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from .config import Settings


def create_app(settings: Settings = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = settings or Settings()

    app = FastAPI(
        title="Progno API",
        description="Scenic route planning powered by AI",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(router, prefix="/api/v1")

    @app.on_event("startup")
    async def startup():
        """Initialize resources on startup."""
        # TODO: Load road graph
        # TODO: Initialize database connection
        # TODO: Load ML models if needed
        pass

    @app.on_event("shutdown")
    async def shutdown():
        """Cleanup on shutdown."""
        pass

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "version": "0.1.0"}

    return app


# Default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
