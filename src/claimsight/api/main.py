"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from claimsight import __version__
from claimsight.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app() -> FastAPI:
    app = FastAPI(
        title="claimsight",
        description="Insurance claims triage: severity prediction with reserve suggestions, fraud-signal flags, complexity-based adjuster routing, and a workload-balanced assignment queue.",
        version=__version__,
    )
    app.include_router(router)
    return app


app = create_app()
