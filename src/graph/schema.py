from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# Graph State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  
    next_node: str                               
    retrieved_context: str  