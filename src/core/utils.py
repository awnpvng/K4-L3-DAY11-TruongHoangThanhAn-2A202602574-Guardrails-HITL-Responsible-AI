"""
Lab 11 — Helper Utilities
"""
import asyncio
import time

from core.config import get_llm_provider, PROVIDER_OPENROUTER  # noqa: F401
from core.openai_runtime import OpenAIRunner


# ---------------------------------------------------------------------------
# Gemini RPM guard.
#
# Free-tier Gemini keys (gemini-3.5-flash) are commonly capped at 5 requests
# per minute; going over risks the key being throttled or disabled. Every
# Gemini call in this lab (smoke test + CP4 attacks on Red + Red Advance)
# goes through chat_with_agent, so we throttle centrally here instead of in
# each call site.
#
# NOTE: google-adk's own SDK silently retries a 429/503 internally *inside*
# a single logical call (several real HTTP requests can happen before it
# gives up or succeeds), so a naive "N calls per 60s" sliding window is not
# safe enough — a burst of logical calls can still blow through the quota
# once retries are counted. We use a fixed minimum spacing between logical
# calls instead (stricter than the stated RPM), and back off further if a
# quota/server error still slips through.
# ---------------------------------------------------------------------------
GEMINI_MAX_REQUESTS_PER_MINUTE = 5
GEMINI_MIN_INTERVAL_SECONDS = 15.0  # ~4 req/min — leaves headroom under the 5 RPM cap
_gemini_last_call_at: float | None = None


async def _throttle_gemini_rpm(min_interval: float = GEMINI_MIN_INTERVAL_SECONDS):
    """Block until at least ``min_interval`` seconds passed since the last Gemini call."""
    global _gemini_last_call_at
    now = time.monotonic()
    if _gemini_last_call_at is not None:
        elapsed = now - _gemini_last_call_at
        wait = min_interval - elapsed
        if wait > 0:
            print(f"  [rate-limit guard] Đợi {wait:.1f}s trước lần gọi Gemini tiếp theo...")
            await asyncio.sleep(wait)
    _gemini_last_call_at = time.monotonic()


def _is_quota_or_overload_error(exc: Exception) -> bool:
    text = str(exc)
    return any(
        marker in text
        for marker in ("RESOURCE_EXHAUSTED", "429", "UNAVAILABLE", "503")
    )


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Send a message to the agent and get the response.

    Works with OpenAIRunner (OpenAI Red / OpenRouter Blue) and Google ADK (Gemini Red).
    """
    provider = getattr(runner, "provider", None)
    if isinstance(runner, OpenAIRunner) or provider in ("openrouter", "openai"):
        text = await runner.chat(agent, user_message)
        return text, None

    # Google ADK path == Gemini Red / Red Advance — throttle to protect the RPM quota.
    await _throttle_gemini_rpm()

    from google.genai import types

    user_id = "student"
    app_name = runner.app_name

    session = None
    if session_id is not None:
        try:
            session = await runner.session_service.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
        except (ValueError, KeyError):
            pass

    if session is None:
        try:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )
        except Exception:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    max_attempts = 3
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            final_response = ""
            async for event in runner.run_async(
                user_id=user_id, session_id=session.id, new_message=content
            ):
                if hasattr(event, "content") and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            final_response += part.text
            return final_response, session
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_attempts and _is_quota_or_overload_error(exc):
                backoff = 20.0 * attempt
                print(
                    f"  [rate-limit guard] Gemini quota/overload (attempt {attempt}/{max_attempts}), "
                    f"chờ {backoff:.0f}s rồi thử lại..."
                )
                await asyncio.sleep(backoff)
                continue
            raise

    raise last_exc  # pragma: no cover — defensive, loop always returns or raises
