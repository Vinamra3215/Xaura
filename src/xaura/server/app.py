"""XAURA FastAPI application factory.

Creates and configures the FastAPI application with:
    - Static file serving (CSS, JS)
    - Jinja2 template engine
    - CORS middleware
    - Route routers (profile, model, export)

Usage:
    from xaura.server.app import create_app
    app = create_app()
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SERVER_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _SERVER_DIR / "templates"
_STATIC_DIR = _SERVER_DIR / "static"


def create_app() -> FastAPI:
    """Create and configure the XAURA FastAPI application.

    Returns:
        A fully configured FastAPI app ready to serve.
    """
    app = FastAPI(
        title="XAURA Dashboard",
        description="eXtendable Automated Unified Research & Analytics",
        version="0.1.0",
    )

    # ── CORS ──────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Static files ──────────────────────────────────────────────────
    _STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # ── Templates ─────────────────────────────────────────────────────
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    # ── In-memory session store ───────────────────────────────────────
    # Stores uploaded DataFrames and profiles between requests.
    # Key: session_id (str), Value: dict with "df", "profile", "result"
    app.state.sessions = {}

    # ── Landing page ──────────────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse(
            name="index.html",
            request=request,
        )

    # ── Run page ──────────────────────────────────────────────────────
    @app.get("/run/{session_id}", response_class=HTMLResponse)
    async def run_page(request: Request, session_id: str):
        session = app.state.sessions.get(session_id, {})
        profile = session.get("profile", None)
        df = session.get("df", None)

        # Get the list of available models
        import xaura.models.classifiers  # noqa: F401
        import xaura.models.clusterers  # noqa: F401
        import xaura.models.regressors  # noqa: F401
        from xaura.models.registry import list_models

        columns = list(df.columns) if df is not None else []
        models = list_models()

        return templates.TemplateResponse(
            name="run.html",
            request=request,
            context={
                "session_id": session_id,
                "profile": profile,
                "columns": columns,
                "models": models,
            },
        )

    # ── Experiments page ──────────────────────────────────────────────
    @app.get("/experiments", response_class=HTMLResponse)
    async def experiments_page(request: Request):
        return templates.TemplateResponse(
            name="experiments.html",
            request=request,
        )

    # ── Include routers ───────────────────────────────────────────────
    # Person A's routes:
    from xaura.server.routes.profile_routes import router as profile_router

    app.include_router(profile_router)

    from xaura.server.routes.model_routes import router as model_router

    app.include_router(model_router)

    # Person B's routes:
    from xaura.server.routes.experiment_routes import router as experiment_router
    from xaura.server.routes.export_routes import router as export_router

    app.include_router(experiment_router)
    app.include_router(export_router)

    return app


# Create a default app instance for uvicorn
app = create_app()
