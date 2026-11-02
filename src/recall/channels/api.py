"""HTTP API for anything that isn't Telegram or MCP.

It listens on localhost, and a reverse proxy should provide HTTPS. Every route sits behind a
bearer token, and all rules live in RecallService.
"""

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from recall.access import Identity
from recall.config import Settings
from recall.factory import build_service
from recall.logs import get_logger, setup
from recall.service import (
    AccessDenied,
    NotFound,
    RateLimited,
    RecallService,
    Unavailable,
)
from recall.storage.base import ToolError

log = get_logger(__name__)

MAX_TEXT = 4000
MAX_SQL = 2000


class SearchRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT)
    table: str | None = None
    limit: int = Field(default=8, ge=1, le=20)


class QueryRequest(BaseModel):
    sql: str = Field(min_length=1, max_length=MAX_SQL)


class RememberRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT)


class ReminderRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT)
    remind_at: str = Field(description="Local time, like 2026-11-06T09:00")
    kind: str = "other"


ERROR_STATUS = {
    AccessDenied: 403,
    NotFound: 404,
    RateLimited: 429,
    ToolError: 400,
    Unavailable: 503,
}


def create_app(service: RecallService) -> FastAPI:
    app = FastAPI(title="Recall", docs_url=None, redoc_url=None, openapi_url=None)
    bearer = HTTPBearer(auto_error=False)

    def identity(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> Identity:
        found = service.tokens.verify(credentials.credentials) if credentials else None
        if found is None:
            log.warning("auth_failed", had_token=credentials is not None)
            raise HTTPException(401, "Invalid or expired token.", {"WWW-Authenticate": "Bearer"})
        return found

    for error, status in ERROR_STATUS.items():

        def handler(_request: Request, exc: Exception, status: int = status) -> JSONResponse:
            return JSONResponse({"detail": str(exc)}, status_code=status)

        app.add_exception_handler(error, handler)

    @app.get("/v1/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/search")
    def search(body: SearchRequest, who: Identity = Depends(identity)) -> dict:
        return {"results": service.search(who, body.text, body.table, body.limit)}

    @app.get("/v1/about/{name}")
    def about(name: str, who: Identity = Depends(identity)) -> dict:
        return {"results": service.about(who, name)}

    @app.post("/v1/query")
    def query(body: QueryRequest, who: Identity = Depends(identity)) -> dict:
        return service.query(who, body.sql)

    @app.get("/v1/tables")
    def tables(who: Identity = Depends(identity)) -> dict:
        return {"tables": service.list_tables(who)}

    @app.get("/v1/entries/{table}/{row_id}")
    def entry(table: str, row_id: int, who: Identity = Depends(identity)) -> dict:
        return service.get_entry(who, table, row_id)

    @app.get("/v1/reminders")
    def reminders(who: Identity = Depends(identity)) -> dict:
        return {"reminders": service.list_reminders(who)}

    @app.post("/v1/reminders")
    def add_reminder(body: ReminderRequest, who: Identity = Depends(identity)) -> dict:
        return service.set_reminder(who, body.text, body.remind_at, body.kind)

    @app.post("/v1/remember")
    def remember(body: RememberRequest, who: Identity = Depends(identity)) -> dict:
        return service.remember(who, body.text)

    return app


def main() -> None:
    settings = Settings.from_env()
    setup(settings.data_dir, "api")
    uvicorn.run(
        create_app(build_service(settings)),
        host="127.0.0.1",
        port=settings.api_port,
        log_config=None,
    )


if __name__ == "__main__":
    main()
