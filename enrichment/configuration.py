### Configuration

from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig, ensure_config
from enrichment import prompts  # See prompts section below


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={
            "description": "The name of the language model to use. Should be in the form: provider/model-name."
        },
    )
    prompt: str = field(
        default=prompts.MAIN_PROMPT,
        metadata={
            "description": "The main prompt template. Expects two arguments: {schema} and {topic}."
        },
    )
    max_loops: int = field(
        default=6,
        metadata={
            "description": "The maximum number of interaction loops before termination."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {
            f.name for f in fields(cls) if f.init
        }  # Gets allowed field names: ["model", "prompt", "max_loops"]
        return cls(
            **{k: v for k, v in configurable.items() if k in _fields}
        )  # Creates Configuration using only matching fields
