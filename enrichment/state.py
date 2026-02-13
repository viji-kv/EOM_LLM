import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

@dataclass(kw_only=True)
class InputState:
    """Initial state passed in by the user."""
    topic: str
    extraction_schema: dict[str, Any]
    info: Optional[dict[str, Any]] = None

@dataclass(kw_only=True)
class State(InputState):
    """Internal state of the agent."""
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)
    loop_step: Annotated[int, operator.add] = field(default=0)

@dataclass(kw_only=True)
class OutputState:
    """Final output delivered to the user."""
    info: dict[str, Any]