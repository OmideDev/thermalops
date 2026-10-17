"""
Experimental informational grounded-assistant HTTP API.

Uses a synchronous route so blocking Ollama HTTP runs in FastAPI's threadpool.
Per-process concurrency is limited to one generation (see AssistantService).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.rag.ollama import OllamaProtocolError, OllamaUnavailableError
from app.schemas.assistant import AssistantAskRequest, AssistantAskResponse
from app.services.assistant import (
    AssistantBusyError,
    AssistantService,
    AssistantUnavailableError,
)

router = APIRouter(
    prefix="/api/v1/assistant",
    tags=["Assistant"],
)


def get_assistant_service(request: Request) -> AssistantService:
    """Resolve the process-scoped assistant; 503 when unavailable."""
    service = getattr(request.app.state, "assistant", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Assistant is unavailable",
        )
    return service


@router.post(
    "/ask",
    response_model=AssistantAskResponse,
    summary="Ask the experimental grounded assistant",
    responses={
        422: {"description": "Invalid question"},
        429: {"description": "Assistant busy (one generation per process)"},
        502: {"description": "Upstream Ollama protocol/invalid response"},
        503: {
            "description": (
                "Assistant disabled/uninitialized, or Ollama unavailable"
            )
        },
        504: {"description": "Upstream Ollama timeout"},
    },
)
def ask_assistant(
    body: AssistantAskRequest,
    service: AssistantService = Depends(get_assistant_service),
) -> AssistantAskResponse:
    """
    Grounded ask over the in-memory eligible-chunk index.

    Synchronous ``def`` so FastAPI schedules the handler in a worker thread,
    keeping the event loop free during blocking Ollama HTTP.
    """
    try:
        result = service.ask(body.question)
    except AssistantBusyError:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "Assistant is busy. This process allows only one generation "
                "at a time."
            ),
        ) from None
    except AssistantUnavailableError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Assistant is unavailable",
        ) from None
    except TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Assistant upstream timed out",
        ) from None
    except OllamaUnavailableError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Assistant upstream is unavailable",
        ) from None
    except ConnectionError:
        # Includes generic connection failures from injected fakes.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Assistant upstream is unavailable",
        ) from None
    except OllamaProtocolError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Assistant upstream returned an invalid response",
        ) from None

    return AssistantAskResponse.model_validate(result)
