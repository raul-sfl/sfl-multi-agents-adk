"""Shared utilities for all Stayforlong agents."""
from google.adk.tools.tool_context import ToolContext


def transfer_to_triage(tool_context: ToolContext) -> dict:
    """Transfer the conversation back to the main Stayforlong assistant for a different topic."""
    tool_context.actions.transfer_to_agent = "Triage"
    return {"status": "transferred"}
