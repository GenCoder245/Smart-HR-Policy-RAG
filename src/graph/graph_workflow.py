from langgraph.graph import StateGraph, START, END
# from langgraph.checkpoint.memory import MemorySaver
from src.graph.schema import AgentState
from src.graph.nodes import set_llm_and_retriever, guardrails, handle_off_topic, handle_unsafe_message, retrieve, generate_answer, routing_after_guardrail


def build_graph(language_model, doc_retriever, memory_checkpointer):

    # To set the LLM Model for the graph nodes to use
    set_llm_and_retriever(language_model, doc_retriever)


    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("guardrails_node", guardrails)
    # graph_builder.add_node("router_node", routing_after_guardrail)
    graph_builder.add_node("off_topic_node", handle_off_topic)
    graph_builder.add_node("unsafe_node",handle_unsafe_message)
    graph_builder.add_node("retrieve_node", retrieve)
    graph_builder.add_node("generate_answer_node", generate_answer)

    graph_builder.add_edge(START, "guardrails_node")
    graph_builder.add_conditional_edges("guardrails_node",
                                        routing_after_guardrail,
                                        {
                                        "retrieve": "retrieve_node",
                                        "off_topic" : "off_topic_node",
                                        "unsafe" : "unsafe_node",
                                        })

    
    graph_builder.add_edge("retrieve_node","generate_answer_node")
    graph_builder.add_edge("generate_answer_node", END)
    graph_builder.add_edge("off_topic_node", END)
    graph_builder.add_edge("unsafe_node", END)

    # memory_checkpointer = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory_checkpointer)

    return graph




    








