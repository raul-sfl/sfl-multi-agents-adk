from dataclasses import dataclass, field
from typing import Callable, Union, Optional, List
import uuid


@dataclass
class Agent:
    name: str
    instructions: Union[str, Callable[[dict], str]]
    tools: List[Callable]
    model: str = ""  # Empty = use GEMINI_MODEL from config.py


@dataclass
class Result:
    """Return type for agent tool functions."""
    value: str = ""
    agent: Optional[Agent] = None
    context_update: dict = field(default_factory=dict)


@dataclass
class Session:
    active_agent: Agent
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    history: List = field(default_factory=list)
    context: dict = field(default_factory=dict)
