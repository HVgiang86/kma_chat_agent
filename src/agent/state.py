from typing import List, Optional, Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class ThinkStep(BaseModel):
    """State of the agent during thinking."""
    index: int = Field(description="The index of the step. It is a number from 1 to n")
    step: str = Field(
        description="The step that you think to do to do the task. It is a string that describes the step")
    step_result: Optional[str] = Field(description="The result of the step. Empty if the step is not done yet")
    is_done: Optional[bool] = Field(
        description="The status of the step. It is a boolean that indicates whether the step is done or not. Default is False")

class InitThinkingOutput(BaseModel):
    """State of the agent during thinking."""
    message: str = Field(description="Some messages you want to talk to user")
    thinking_steps: Optional[List[ThinkStep]] = Field(description="The steps that you think to do to do the task")

class RePlanningOutput(BaseModel):
    """State of the agent during thinking."""
    message: str = Field(description="Some messages you want to talk to user")
    need_replan: bool = Field(description="Indicate whether you need to replan or not")
    thinking_steps: Optional[List[ThinkStep]] = Field(
        description="The steps after re-plan that you think to do to do the task. Leave this empty if dont need to replan")

class DoTaskOutput(BaseModel):
    """State of the agent during do -task."""
    message: str = Field(description="The result of tasks you did. And something you want to talk to user")
    completed_steps: Optional[List[int]] = Field(
        description="The steps 's indexs that you have done to take result. Leave this empty if no step is done yet")
    has_final_result: bool = Field(
        description="Indicate whether you have final result or not. Default is False. If True, you have final result and no need to do any step anymore")

# --- Định nghĩa State ---
class MyAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    thinking_steps: Optional[List[ThinkStep]]
    previous_thinking: Optional[List[ThinkStep]]
    finish: Optional[bool]
