import asyncio
import inspect
import logging
import time

import google.generativeai as genai
from google.generativeai import protos

from config import GEMINI_MODEL, GENERATION_CONFIG
from orchestrator.types import Agent, Result, Session
from orchestrator.utils import build_gemini_tools

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 6
MAX_RETRIES = 3
RETRY_BASE_DELAY = 10  # seconds — Gemini free tier resets per minute


def resolve_instructions(agent: Agent, context: dict) -> str:
    if callable(agent.instructions):
        return agent.instructions(context)
    return agent.instructions


def extract_function_calls(response) -> list:
    try:
        parts = response.candidates[0].content.parts
        return [p for p in parts if p.function_call.name]
    except Exception:
        return []


def extract_text(response) -> str:
    try:
        parts = response.candidates[0].content.parts
        texts = [p.text for p in parts if hasattr(p, "text") and p.text]
        return "\n".join(texts).strip()
    except Exception:
        return ""


def execute_tool(function_call, tools: list, context: dict) -> Result:
    name = function_call.name
    args = dict(function_call.args) if function_call.args else {}

    func = next((f for f in tools if f.__name__ == name), None)
    if func is None:
        logger.warning(f"Tool '{name}' not found.")
        return Result(value=f"Error: tool '{name}' not found.")

    sig = inspect.signature(func)
    if "context_variables" in sig.parameters:
        args["context_variables"] = context

    try:
        raw = func(**args)
    except Exception as e:
        logger.error(f"Tool '{name}' raised: {e}")
        return Result(value=f"Error executing {name}: {str(e)}")

    if isinstance(raw, Result):
        return raw
    if isinstance(raw, Agent):
        return Result(
            value=(
                f"Transfer to {raw.name} complete. "
                "Continue the conversation naturally. "
                "Do NOT greet or introduce yourself — the user experiences this as one unified assistant. "
                "You may briefly note the topic area only if it genuinely adds clarity, "
                "then immediately address the user's request."
            ),
            agent=raw,
        )
    return Result(value=str(raw))


def user_content(text: str) -> protos.Content:
    return protos.Content(role="user", parts=[protos.Part(text=text)])


def tool_results_content(parts: list) -> protos.Content:
    return protos.Content(role="user", parts=parts)


def function_response_part(name: str, value: str) -> protos.Part:
    return protos.Part(
        function_response=protos.FunctionResponse(
            name=name,
            response={"result": value},
        )
    )


def _is_rate_limit_error(e: Exception) -> bool:
    """Detect 429 / quota-exceeded errors from the Gemini API."""
    msg = str(e).lower()
    return (
        "429" in msg
        or "quota" in msg
        or "resource_exhausted" in msg
        or "resourceexhausted" in msg
        or type(e).__name__ in ("ResourceExhausted", "TooManyRequests")
    )


def _generate_sync(model, history):
    return model.generate_content(history)


async def _generate_with_retry(model, history) -> object:
    """Call Gemini with exponential backoff on rate-limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return await asyncio.to_thread(_generate_sync, model, history)
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)  # 10s, 20s, 40s
                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{MAX_RETRIES}). "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
            else:
                raise  # re-raise on last attempt or non-rate-limit errors


async def run_turn(session: Session, user_message: str) -> str:
    session.history.append(user_content(user_message))

    for round_num in range(MAX_TOOL_ROUNDS):
        instructions = resolve_instructions(session.active_agent, session.context)
        tools = build_gemini_tools(session.active_agent.tools)
        model_name = session.active_agent.model or GEMINI_MODEL

        logger.info(f"[Round {round_num}] Agent={session.active_agent.name} model={model_name}")

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=instructions,
            tools=tools,
            generation_config=GENERATION_CONFIG,
        )

        try:
            response = await _generate_with_retry(model, session.history)
        except Exception as e:
            logger.error(f"Gemini API error: {type(e).__name__}: {e}")
            if _is_rate_limit_error(e):
                return (
                    "The API rate limit has been reached. Please wait a moment and try again.\n"
                    "Se ha alcanzado el límite de peticiones. Por favor espera un momento e inténtalo de nuevo."
                )
            return f"[Gemini error] {type(e).__name__}: {e}"

        try:
            session.history.append(response.candidates[0].content)
        except (IndexError, AttributeError) as e:
            logger.error(f"Bad response structure: {e} | {response}")
            return f"[Response error] {e}"

        function_calls = extract_function_calls(response)

        if not function_calls:
            text = extract_text(response)
            return text if text else "No response generated."

        tool_result_parts = []
        for fc in function_calls:
            result = execute_tool(fc.function_call, session.active_agent.tools, session.context)
            if result.agent:
                logger.info(f"Handoff: {session.active_agent.name} -> {result.agent.name}")
                session.active_agent = result.agent
            session.context.update(result.context_update)
            tool_result_parts.append(function_response_part(fc.function_call.name, result.value))

        session.history.append(tool_results_content(tool_result_parts))

    return "Could not complete the request. Please try again."
