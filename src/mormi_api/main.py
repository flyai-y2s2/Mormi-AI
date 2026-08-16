from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from .db import Database
from .dictionary_catalog import (
    DictionaryCardNotFoundError,
    DictionaryVersionMismatchError,
    dictionary_card_envelope,
    get_dictionary_card,
    validate_dictionary_catalog,
)
from .dictionary_models import DictionaryCardEnvelope
from .engine import ConversationEngine
from .llm import ClaudeGateway, ModelOutputError, ModelUnavailableError
from .repository import (
    ConversationNotFoundError,
    Repository,
    StaleConversationError,
)
from .schemas import (
    ChildResponse,
    ConflictDetail,
    ConflictResponse,
    HealthResponse,
    PracticeResult,
    SessionCreate,
    SessionEnvelope,
    SkillProfilesResponse,
    StarNotesResponse,
)
from .security import TextCipher
from .service import ConversationService
from .settings import Settings, get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Trusted reference content is a startup contract, not a best-effort
    # runtime fallback. Invalid, stale or incomplete catalogs cannot be served.
    validate_dictionary_catalog()
    settings = get_settings()
    database = Database(settings.database_url)
    await database.create_schema()
    gateway = ClaudeGateway(settings)
    repository = Repository(
        database,
        TextCipher(settings.raw_data_encryption_key),
        idempotency_retention_days=settings.idempotency_retention_days,
    )
    await repository.migrate_existing_storage_to_permanent()
    await repository.purge_expired_raw_data()
    engine = ConversationEngine(
        gateway,
        show_internal_pedagogy=settings.show_internal_pedagogy,
        speaker_timeout_seconds=settings.speaker_timeout_seconds,
        semantic_verifier_enabled=settings.speaker_verifier_enabled,
        verifier_timeout_seconds=settings.speaker_verifier_timeout_seconds,
    )
    app.state.settings = settings
    app.state.database = database
    app.state.gateway = gateway
    app.state.repository = repository
    app.state.service = ConversationService(repository, engine)
    yield
    await database.dispose()


app = FastAPI(
    title="Mormi Conversation API",
    version="0.1.0",
    description=(
        "집 반복 결과를 받아 모르미 가르치기와 카페 생활수학 대화를 진행하는 LangGraph 기반 API"
    ),
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Mormi-Service-Key"],
)


def service(request: Request) -> ConversationService:
    return request.app.state.service  # type: ignore[no-any-return]


def repository(request: Request) -> Repository:
    return request.app.state.repository  # type: ignore[no-any-return]


def require_service_key(
    request: Request,
    key: Annotated[str | None, Header(alias="X-Mormi-Service-Key")] = None,
) -> None:
    current: Settings = request.app.state.settings
    if current.service_api_key and key != current.service_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid service key")


Auth = Annotated[None, Depends(require_service_key)]
Service = Annotated[ConversationService, Depends(service)]
Repo = Annotated[Repository, Depends(repository)]

ERROR_CODE_HEADER = "X-Mormi-Error-Code"
ERROR_PATH_HEADER = "X-Mormi-Error-Path"


def _validation_issues(error: RequestValidationError | ValidationError) -> list[dict[str, str]]:
    """Return field paths and rule names without echoing request values.

    FastAPI's default 422 body includes the rejected ``input``. That is useful
    for ordinary APIs, but this service can receive a child's utterance on
    other endpoints. Operational diagnostics therefore expose only the schema
    location and validation rule.
    """

    issues: list[dict[str, str]] = []
    for issue in error.errors()[:5]:
        location = ".".join(str(part) for part in issue.get("loc", ())) or "request"
        issues.append(
            {
                "location": location[:160],
                "type": str(issue.get("type", "value_error"))[:80],
            }
        )
    return issues or [{"location": "request", "type": "value_error"}]


def _diagnostic_headers(code: str, path: str | None = None) -> dict[str, str]:
    headers = {ERROR_CODE_HEADER: code[:80]}
    if path:
        # Pydantic locations contain field names and numeric list indexes only.
        headers[ERROR_PATH_HEADER] = path[:160]
    return headers


def _request_validation_code(error: RequestValidationError) -> str:
    known_codes = (
        ("consented raw storage requires", "storage_consent_retention_mismatch"),
        ("retention_policy must be no_raw", "storage_retention_without_consent"),
        ("learning_session_id is required", "home_learning_session_missing"),
        ("practice_result_id is required", "home_practice_result_missing"),
        ("cafe_context is required", "cafe_context_missing"),
        ("cafe_context is not used", "cafe_context_unexpected"),
        ("queue_context is required", "queue_context_missing"),
        ("queue_context is not used", "queue_context_unexpected"),
        ("budget is required", "cafe_budget_missing"),
    )
    messages = " ".join(str(issue.get("msg", "")) for issue in error.errors())
    for fragment, code in known_codes:
        if fragment in messages:
            return code
    first_type = (
        str(error.errors()[0].get("type", "value_error"))
        if error.errors()
        else "value_error"
    )
    if first_type.replace("_", "").replace(".", "").isalnum():
        return f"request_validation_failed.{first_type}"[:80]
    return "request_validation_failed"


def _service_error_code(error: ValueError) -> tuple[str, str | None, list[dict[str, str]]]:
    if isinstance(error, ValidationError):
        issues = _validation_issues(error)
        return "stored_state_validation_failed", issues[0]["location"], issues

    message = str(error)
    known_codes = (
        ("unsupported home curriculum_session_id", "home_curriculum_unsupported"),
        ("curriculum_session_id is required", "home_curriculum_missing"),
        ("practice result does not belong", "practice_result_owner_mismatch"),
        ("practice_result_id is unavailable", "practice_result_unavailable"),
        ("practice_summary or practice_result_id is required", "practice_summary_missing"),
        ("scenario_id does not belong", "scenario_scene_mismatch"),
        ("cafe_context is required", "cafe_context_missing"),
        ("queue_context is required", "queue_context_missing"),
    )
    for fragment, code in known_codes:
        if fragment in message:
            return code, None, []
    return "conversation_configuration_invalid", None, []


def _model_error_code(error: ModelUnavailableError | ModelOutputError) -> str:
    allowed = {
        "model_connection_failed",
        "model_bad_request",
        "model_auth_failed",
        "model_forbidden",
        "model_not_found",
        "model_rate_limited",
        "model_provider_unavailable",
        "structured_schema_not_strict",
        "structured_schema_too_complex",
        "structured_schema_invalid",
    }
    code = str(error)
    if code in allowed:
        return code
    return "model_output_invalid" if isinstance(error, ModelOutputError) else "model_unavailable"


@app.exception_handler(RequestValidationError)
async def request_validation_error(
    _: Request,
    error: RequestValidationError,
) -> JSONResponse:
    issues = _validation_issues(error)
    code = _request_validation_code(error)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": {"code": code, "issues": issues}},
        headers=_diagnostic_headers(code, issues[0]["location"]),
    )


@app.get("/health", response_model=HealthResponse, tags=["operations"])
async def health(request: Request) -> HealthResponse:
    current: Settings = request.app.state.settings
    gateway: ClaudeGateway = request.app.state.gateway
    return HealthResponse(
        llm_configured=gateway.configured,
        database="postgresql" if current.database_url.startswith("postgresql") else "sqlite",
    )


@app.get(
    "/health/authenticated",
    response_model=HealthResponse,
    tags=["operations"],
)
async def authenticated_health(request: Request, _: Auth) -> HealthResponse:
    """Verify both service reachability and the BE-to-AI shared key."""

    return await health(request)


@app.post(
    "/v1/practice-results",
    response_model=PracticeResult,
    status_code=status.HTTP_201_CREATED,
    tags=["practice integration"],
)
async def save_practice_result(
    body: PracticeResult,
    _: Auth,
    conversation: Service,
) -> PracticeResult:
    return await conversation.save_practice(body)


@app.get(
    "/v1/content/dictionary-cards/{curriculum_session_id}",
    response_model=DictionaryCardEnvelope,
    responses={
        404: {"description": "No reviewed dictionary card is registered."},
        409: {"description": "The caller expected another content version."},
    },
    tags=["dictionary content"],
)
async def read_dictionary_card(
    curriculum_session_id: str,
    _: Auth,
    expected_content_version: int | None = None,
) -> DictionaryCardEnvelope:
    """Read one approved catalog card without invoking an LLM."""

    try:
        card = get_dictionary_card(
            curriculum_session_id,
            expected_content_version=expected_content_version,
        )
    except DictionaryCardNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail={"code": "dictionary_card_not_found"},
        ) from error
    except DictionaryVersionMismatchError as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "dictionary_version_mismatch",
                "expected_content_version": error.expected,
                "current_content_version": error.actual,
            },
        ) from error
    return dictionary_card_envelope(card)


@app.get(
    "/v1/conversations/{conversation_id}/dictionary-card",
    response_model=DictionaryCardEnvelope,
    responses={
        404: {"description": "Conversation not found."},
        409: {"description": "The legacy conversation has no pinned card."},
    },
    tags=["dictionary content"],
)
async def read_conversation_dictionary_card(
    conversation_id: str,
    _: Auth,
    conversation: Service,
) -> DictionaryCardEnvelope:
    """Read the exact card snapshot pinned to the conversation's active task."""

    try:
        return await conversation.dictionary_card(conversation_id)
    except ConversationNotFoundError as error:
        raise HTTPException(status_code=404, detail="Conversation not found") from error
    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail={"code": "dictionary_snapshot_unavailable"},
        ) from error


@app.post(
    "/v1/conversations",
    response_model=SessionEnvelope,
    status_code=status.HTTP_201_CREATED,
    tags=["conversation"],
)
async def create_conversation(
    body: SessionCreate,
    _: Auth,
    conversation: Service,
) -> SessionEnvelope:
    try:
        return await conversation.create_conversation(body)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        code, path, issues = _service_error_code(error)
        raise HTTPException(
            status_code=422,
            detail={"code": code, "issues": issues},
            headers=_diagnostic_headers(code, path),
        ) from error


@app.post(
    "/v1/conversations/{conversation_id}/responses",
    response_model=SessionEnvelope,
    responses={
        409: {
            "model": ConflictResponse,
            "description": "The response targeted a stale or already answered turn.",
        }
    },
    tags=["conversation"],
)
async def respond(
    conversation_id: str,
    body: ChildResponse,
    _: Auth,
    conversation: Service,
) -> SessionEnvelope:
    try:
        return await conversation.respond(conversation_id, body)
    except ConversationNotFoundError as error:
        raise HTTPException(status_code=404, detail="Conversation not found") from error
    except (StaleConversationError, ValueError) as error:
        try:
            latest = await conversation.snapshot(conversation_id)
        except ConversationNotFoundError:
            raise HTTPException(status_code=404, detail="Conversation not found") from error
        detail = ConflictDetail(
            message=str(error),
            conversation_id=conversation_id,
            turn_id=latest.turn.turn_id,
            state_version=latest.turn.state_version,
        )
        raise HTTPException(status_code=409, detail=detail.model_dump(mode="json")) from error
    except (ModelUnavailableError, ModelOutputError) as error:
        code = _model_error_code(error)
        raise HTTPException(
            status_code=503,
            detail={"code": code, "issues": []},
            headers=_diagnostic_headers(code),
        ) from error


def _sse(event: str, data: dict[str, object], *, event_id: str | None = None) -> str:
    lines = [f"event: {event}"]
    if event_id:
        lines.append(f"id: {event_id}")
    lines.append(f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}")
    return "\n".join(lines) + "\n\n"


def _text_chunks(text: str, size: int = 8) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)]


@app.post(
    "/v1/conversations/{conversation_id}/responses/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "Progress, validated Mormi text deltas, and the authoritative turn.",
        }
    },
    tags=["conversation"],
)
async def respond_stream(
    conversation_id: str,
    body: ChildResponse,
    _: Auth,
    conversation: Service,
) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        try:
            async for event in conversation.respond_stream(conversation_id, body):
                if event.name == "accepted":
                    yield _sse(
                        "response.accepted",
                        {
                            "conversation_id": conversation_id,
                            "response_id": str(body.response_id),
                        },
                    )
                    continue
                if event.name == "progress" and event.stage:
                    yield _sse(
                        "response.progress",
                        {"conversation_id": conversation_id, "stage": event.stage},
                    )
                    continue
                if event.envelope is None:
                    continue

                envelope = event.envelope
                turn_id = envelope.turn.turn_id
                yield _sse(
                    "mormi.start",
                    {"turn_id": turn_id, "replayed": event.replayed},
                    event_id=turn_id,
                )
                for delta in _text_chunks(envelope.turn.mormi.text):
                    yield _sse("mormi.delta", {"turn_id": turn_id, "delta": delta})
                    # Yield control so ASGI can flush each already-validated chunk.
                    await asyncio.sleep(0)
                yield _sse(
                    "turn.completed",
                    envelope.model_dump(mode="json"),
                    event_id=turn_id,
                )
                yield _sse("done", {"turn_id": turn_id})
        except ConversationNotFoundError:
            yield _sse(
                "error",
                {"code": "conversation_not_found", "retryable": False},
            )
        except (StaleConversationError, ValueError):
            yield _sse(
                "error",
                {"code": "stale_or_invalid_response", "retryable": False},
            )
        except (ModelUnavailableError, ModelOutputError) as error:
            yield _sse(
                "error",
                {"code": _model_error_code(error), "retryable": True},
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get(
    "/v1/conversations/{conversation_id}",
    response_model=SessionEnvelope,
    tags=["conversation"],
)
async def get_conversation(
    conversation_id: str,
    _: Auth,
    conversation: Service,
) -> SessionEnvelope:
    try:
        return await conversation.snapshot(conversation_id)
    except ConversationNotFoundError as error:
        raise HTTPException(status_code=404, detail="Conversation not found") from error


@app.get(
    "/v1/learners/{learner_id}/skill-profiles",
    response_model=SkillProfilesResponse,
    tags=["learner state"],
)
async def get_learner_profiles(
    learner_id: int,
    _: Auth,
    repo: Repo,
) -> SkillProfilesResponse:
    profile = await repo.get_profile(learner_id)
    return SkillProfilesResponse(learner_id=learner_id, skills=list(profile.skills.values()))


@app.get(
    "/v1/learners/{learner_id}/star-notes",
    response_model=StarNotesResponse,
    tags=["learner state"],
)
async def get_star_notes(learner_id: int, _: Auth, repo: Repo) -> StarNotesResponse:
    return StarNotesResponse(learner_id=learner_id, notes=await repo.list_notes(learner_id))


@app.get(
    "/v1/conversations/{conversation_id}/transcript",
    tags=["protected dialogue"],
)
async def get_transcript(
    conversation_id: str,
    _: Auth,
    repo: Repo,
) -> list[dict[str, object]]:
    try:
        return await repo.raw_turns(conversation_id)
    except ConversationNotFoundError as error:
        raise HTTPException(status_code=404, detail="Conversation not found") from error


def run() -> None:
    uvicorn.run("mormi_api.main:app", host="0.0.0.0", port=8000, reload=True)
