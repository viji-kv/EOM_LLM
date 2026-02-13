import json
from typing import Any, Dict, List, Optional, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from enrichment.configuration import Configuration
from enrichment import prompts
from enrichment.state import InputState, OutputState, State
from enrichment.utils import init_model


async def call_agent_model(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Call the LLM to produce an answer based on the provided topic and extraction schema.
    Since we're not using external tools, we simply format the prompt and pass in the conversation history.
    """
    print("---call_agent_model step---")
    configuration = Configuration.from_runnable_config(config)

    # Format the prompt with the extraction schema and topic.
    prompt_text = configuration.prompt.format(
        info=json.dumps(state.extraction_schema, indent=2), topic=state.topic
    )
    # Start with the new prompt message, then include any prior messages.
    messages: List[BaseMessage] = [HumanMessage(content=prompt_text)] + state.messages

    # Initialize and call the model.
    model = init_model(config)
    response = cast(AIMessage, await model.ainvoke(messages))

    # In this simplified version, we take the LLM's response as the final info.
    info = response.content

    new_messages = state.messages + [response]
    return {
        "messages": new_messages,
        "info": info,
        "loop_step": state.loop_step + 1,
    }


# A simplified reflection step that asks the LLM if the answer is satisfactory.
class ReflectionResult(BaseModel):
    is_satisfactory: bool = Field(
        ..., description="Whether the answer is satisfactory."
    )
    feedback: str = Field(..., description="Feedback on the answer.")


async def reflect(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Use the LLM to review the answer.
    The model is asked to confirm if the provided answer is satisfactory.
    """
    print("---Reflect step---")
    prompt_text = (
        "Review the following answer and determine if it adequately addresses the topic.\n\n"
        f"Answer:\n{state.info}\n\n"
        "Respond with 'Yes' if the answer is satisfactory; otherwise, include feedback and suggestions."
    )
    messages: List[BaseMessage] = [HumanMessage(content=prompt_text)]
    model = init_model(config).with_structured_output(ReflectionResult)
    reflection: ReflectionResult = await model.ainvoke(messages)

    # If the answer is satisfactory, we finish. Otherwise, we can loop for improvement.
    if reflection.is_satisfactory:
        return {
            "info": state.info,
            "messages": [HumanMessage(content=reflection.feedback)],
        }
    else:
        return {
            "messages": [HumanMessage(content=f"Feedback: {reflection.feedback}")],
        }


### Routing Functions


def route_after_agent(state: State) -> str:
    """
    Decide the next step.
    If an answer is produced, move to reflection.
    """
    print("---Route_after_agent step---")
    if state.info:
        return "reflect"
    return "call_agent_model"


def route_after_checker(state: State, config: RunnableConfig) -> str:
    """
    Decide whether to iterate or finish.
    If the loop count is below max_loops and the answer is unsatisfactory, repeat.
    Otherwise, terminate.
    """
    configuration = Configuration.from_runnable_config(config)
    if state.loop_step < configuration.max_loops and not state.info:
        return "call_agent_model"
    return "__end__"


### Workflow Graph

# Create the graph with just the two main nodes.
workflow = StateGraph(
    State, input=InputState, output=OutputState, config_schema=Configuration
)
workflow.add_node(call_agent_model)
workflow.add_node(reflect)
workflow.add_edge("__start__", "call_agent_model")
workflow.add_conditional_edges("call_agent_model", route_after_agent)
workflow.add_conditional_edges("reflect", route_after_checker)

graph = workflow.compile()
graph.name = "ResearchTopic"

graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
