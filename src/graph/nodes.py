from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.prompt.prompts import SYSTEM_PROMPT, GUARDRAIL_PROMPT, OFF_TOPIC_MESSAGE, UNSAFE_MESSAGE, FALLBACK_MESSAGE
from src.graph.schema import AgentState
from langchain_core.tools import tool
from src.retrieval.retriever import PolicyRetriever
from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm_model: ChatOpenAI | ChatGoogleGenerativeAI
retriever: PolicyRetriever

# To set the LLM Model for the graph nodes to use
def set_llm_and_retriever(language_model, doc_retriever):
    global llm_model, retriever
    # llm_model = language_model | StrOutputParser()
    llm_model = language_model
    retriever = doc_retriever


@tool
def hr_policy_documents_search(search_query: str):
    """
    Search the HR Policy documents for relevant information.
    Used for searching the workplace HR policy documents for user questions such as but not limited to 
    attendance, leave, remote work, performance, benefits, internal movement etc
    """
     
    # retrieved_docs = retriever.search(search_query)
    retrieved_docs = retriever.retrieve_documents(search_query)
    if not retrieved_docs:
        return "NO_CONTEXT"

    reranked_documents = retriever.rerank_documents(query = search_query, 
                               ranked_docs = retrieved_docs)
    
    return retriever.format_documents(reranked_documents)


def guardrails(state: AgentState):
    last_msg = state["messages"][-1].content

    messages = [
                SystemMessage(content=(GUARDRAIL_PROMPT)),
                HumanMessage(content=last_msg),
            ]
    response = llm_model.invoke(messages)

    label = response.text.strip().upper()
        
    if "UNSAFE" in label:
        return {"next_node": "unsafe"}
    if "OFF_TOPIC" in label or "OFF-TOPIC" in label:
        return {"next_node": "off_topic"}
    if "SAFE" in label:
        return {"next_node": "safe"}


def routing_after_guardrail(state: AgentState):
    return "retrieve" if state["next_node"] == "safe" else state["next_node"]


def handle_off_topic(state: AgentState):
    return {"messages": [AIMessage(content=OFF_TOPIC_MESSAGE)]}


def handle_unsafe_message(state:AgentState):
    return {"messages" : [AIMessage(content = UNSAFE_MESSAGE)]}


def retrieve(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1].content

    # Prior chat_history exists. Therefore need to rephrase the query to be a 
    # stand-alone query based on the entire chat history.
    if len(messages) > 1:
        rephrased_chat_history = messages + [
            SystemMessage(content=(
                "Given the user's chat history and the latest user question, rewrite the "
                "question accordingly so it is fully self-contained (resolve pronouns like 'she', "
                "'it', 'that' etc).  ONLY RETURN THE RE-WRITTEN STANDALONE QUESTION."
                "DO NOT ANSWER THE QUESTION YOURSELF."
            )),
        ]
        standalone_query = llm_model.invoke(rephrased_chat_history).content.strip()

        # The tool will take care of formatting the documents and return only the context for user_query.
        # context = hr_policy_documents_search.invoke({"query": standalone_query})
        context = hr_policy_documents_search.invoke(input=standalone_query)
        

    else:  
        # The tool will take care of formatting the documents and return only the context for user_query.
        context = hr_policy_documents_search.invoke(input=last_message)

    # This just retrieved the context, it doesn't contain the answer to the user's latest query yet.
    # Also this node doesn't add to existing messages list.
    return {"retrieved_context": context}



def generate_answer(state: AgentState):
    context = state.get("retrieved_context", "")
    if not context:
        return {"messages": [AIMessage(content=FALLBACK_MESSAGE)]}

    formatted_prompt = SYSTEM_PROMPT.format(context = context)
    
    full_messages = [SystemMessage(content = formatted_prompt)] + state["messages"]

    response = llm_model.invoke(full_messages)

    return {"messages": [response]}
