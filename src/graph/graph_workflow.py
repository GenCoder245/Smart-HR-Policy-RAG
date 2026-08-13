from langgraph.graph import StateGraph, START, END
# from langgraph.checkpoint.memory import MemorySaver
from src.graph.schema import AgentState
from src.graph.nodes import (
    set_llm_and_retriever,
    guardrails,
    handle_greeting,
    handle_off_topic,
    handle_unsafe_message,
    retrieve,
    generate_answer,
    routing_after_guardrail,
    conversation_follow_up_check,
    route_after_conversation_check,
    answer_from_history,
)


def build_graph(language_model, doc_retriever, memory_checkpointer):

    # To set the LLM Model for the graph nodes to use
    set_llm_and_retriever(language_model, doc_retriever)

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("guardrails_node", guardrails)
    graph_builder.add_node("greeting_node", handle_greeting)
    graph_builder.add_node("conversation_check_node", conversation_follow_up_check)
    graph_builder.add_node("answer_from_history_node", answer_from_history)
    graph_builder.add_node("off_topic_node", handle_off_topic)
    graph_builder.add_node("unsafe_node", handle_unsafe_message)
    graph_builder.add_node("retrieve_node", retrieve)
    graph_builder.add_node("generate_answer_node", generate_answer)

    graph_builder.add_edge(START, "guardrails_node")
    graph_builder.add_conditional_edges(
        "guardrails_node",
        routing_after_guardrail,
        {
            "greeting": "greeting_node",
            "conversation_check": "conversation_check_node",
            "off_topic": "off_topic_node",
            "unsafe": "unsafe_node",
        },
    )
    graph_builder.add_conditional_edges(
        "conversation_check_node",
        route_after_conversation_check,
        {
            "retrieve": "retrieve_node",
            "answer_from_history": "answer_from_history_node",
        },
    )

    graph_builder.add_edge("retrieve_node", "generate_answer_node")
    graph_builder.add_edge("generate_answer_node", END)
    graph_builder.add_edge("answer_from_history_node", END)
    graph_builder.add_edge("greeting_node", END)
    graph_builder.add_edge("off_topic_node", END)
    graph_builder.add_edge("unsafe_node", END)

    # memory_checkpointer = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory_checkpointer)

    return graph




    








