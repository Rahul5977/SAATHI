"""
Chat transports.

Three endpoints:
  - WebSocket `/ws/{session_id}`     — primary streaming transport
  - REST     `/chat/sync`            — non-streaming convenience endpoint
  - REST     `/session/{id}/state`   — debug / observability endpoint

The WebSocket protocol (server → client) is one of:
  {"type": "typing"}                            — model is generating
  {"type": "token", "content": "<piece>"}       — one stream chunk
  {"type": "done",  "meta": {...}}              — turn complete
  {"type": "error", "content": "<msg>"}         — recoverable error

Client sends:
  {"message": "<seeker text>"}                  — one turn

A SINGLE PipelineOrchestrator is shared across requests. It owns the FAISS
index (~100 MB resident) and any open Redis connection — recreating it per
request would be catastrophic for cold-start latency.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from pipeline.orchestrator import PipelineOrchestrator


logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level singleton — cold start (FAISS load) happens here exactly once.
orchestrator = PipelineOrchestrator()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _session_to_meta(session) -> dict[str, Any]:
    """Project a SessionState into the small dict we ship to the client.
    Returns {} if session is None. Never raises."""
    meta: dict[str, Any] = {}
    if session is None:
        return meta

    sd = session.latest_strategy_decision
    if sd is not None:
        meta["phase"] = sd.current_phase
        meta["strategy"] = sd.selected_strategy
        meta["lens"] = sd.restatement_lens

    if session.latest_phase_decision_reason:
        meta["phase_decision"] = session.latest_phase_decision_reason

    az = session.latest_analyzer_state
    if az is not None:
        meta["emotion"] = az.emotion_type
        meta["intensity"] = az.emotion_intensity
        meta["coping"] = az.current_coping_mech
        meta["coping_shade"] = az.coping_shade_signal
        meta["receptiveness"] = az.user_receptiveness
        meta["stigma_cue"] = az.stigma_cue
        meta["risk_signal"] = az.risk_signal

    sf = session.latest_safety_flags
    if sf is not None:
        meta["safety_risk"] = sf.risk_level
        meta["safety_trigger"] = sf.trigger_phrase

    meta["turn_count"] = session.turn_count
    meta["intensity_trajectory"] = session.intensity_trajectory[-10:]
    meta["phase_history"]        = session.phase_history[-10:]
    meta["strategy_history"]     = session.strategy_history[-10:]

    # ---- memory layer surfaces (debug panel) ------------------------------
    meta["turns_in_current_phase"] = session.turns_in_current_phase
    meta["phase_first_reached"]    = session.phase_first_reached
    meta["facts_log_count"]        = len(session.facts_log)
    meta["facts_log_recent"]       = session.facts_log[-6:]
    if session.summary is not None:
        meta["summary_at_turn"] = session.summary.generated_at_turn
        meta["seeker_goal"]     = session.summary.seeker_goal
        meta["emotional_arc"]   = session.summary.emotional_arc
        meta["phase_journey"]   = session.summary.phase_journey
        meta["open_threads"]    = session.summary.open_threads
    if session.user_profile_snapshot is not None:
        prof = session.user_profile_snapshot
        meta["user_sessions_count"] = prof.sessions_count
        meta["user_key_life_facts_count"] = len(prof.key_life_facts)
    return meta


def _user_id_for(session_id: str) -> str:
    """Derive a stable user_id from session_id when the client doesn't
    supply one. Replace once we have real auth."""
    return f"user_{session_id[:8]}"


# ---------------------------------------------------------------------------
# WebSocket — primary streaming transport
# ---------------------------------------------------------------------------
@router.websocket("/ws/{session_id}")
async def chat_websocket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    user_id = _user_id_for(session_id)
    logger.info("WS connected: session=%s user=%s", session_id, user_id)

    try:
        while True:
            # ---- receive one turn from the client ----
            data = await websocket.receive_json()
            seeker_text = (data.get("message") or "").strip()
            if not seeker_text:
                await websocket.send_json(
                    {"type": "error", "content": "Empty message"}
                )
                continue

            # ---- typing indicator ----
            await websocket.send_json({"type": "typing"})

            # ---- stream the orchestrator's tokens ----
            try:
                async for token in orchestrator.run(
                    session_id, user_id, seeker_text
                ):
                    await websocket.send_json(
                        {"type": "token", "content": token}
                    )
            except Exception as e:
                # Orchestrator's internal agents already swallow most failures;
                # this only fires on truly unexpected blow-ups.
                logger.error(
                    "Orchestrator failed mid-stream (session=%s): %s",
                    session_id, e, exc_info=True,
                )
                await websocket.send_json(
                    {"type": "error", "content": "Internal error during generation"}
                )
                # Move on to next turn rather than killing the socket.
                continue

            # ---- send completion + per-turn metadata ----
            session = await orchestrator.session_manager.get_session(session_id)
            await websocket.send_json(
                {"type": "done", "meta": _session_to_meta(session)}
            )

    except WebSocketDisconnect:
        logger.info("WS disconnected: session=%s", session_id)
    except Exception as e:
        logger.error(
            "WS handler crashed (session=%s): %s", session_id, e, exc_info=True,
        )
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            # Socket may already be closed.
            pass


# ---------------------------------------------------------------------------
# REST — non-streaming convenience endpoint
# ---------------------------------------------------------------------------
class SyncChatRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None,
        description="Existing session id; a new one is generated if omitted.",
    )
    message: str = Field(min_length=1, description="The seeker's message.")


class SyncChatResponse(BaseModel):
    session_id: str
    response: str
    meta: dict


@router.post("/chat/sync", response_model=SyncChatResponse)
async def chat_sync(request: SyncChatRequest) -> SyncChatResponse:
    """Non-streaming variant. Useful for curl / smoke tests / batch eval."""
    session_id = request.session_id or str(uuid.uuid4())
    user_id = _user_id_for(session_id)

    # Drain the orchestrator stream into a single string.
    parts: list[str] = []
    try:
        async for token in orchestrator.run(
            session_id, user_id, request.message.strip()
        ):
            parts.append(token)
    except Exception as e:
        logger.error("chat_sync orchestrator failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Pipeline error") from e

    full_response = "".join(parts).strip()
    session = await orchestrator.session_manager.get_session(session_id)
    return SyncChatResponse(
        session_id=session_id,
        response=full_response,
        meta=_session_to_meta(session),
    )


# ---------------------------------------------------------------------------
# REST — debug / observability
# ---------------------------------------------------------------------------
@router.get("/session/{session_id}/state")
async def get_session_state(session_id: str) -> dict:
    """Dump the full SessionState as JSON. Used by the debug UI panel."""
    session = await orchestrator.session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.model_dump(mode="json")
