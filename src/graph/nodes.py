import re

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.prompt.prompts import (
    SYSTEM_PROMPT,
    GUARDRAIL_PROMPT,
    OFF_TOPIC_MESSAGE,
    UNSAFE_MESSAGE,
    FALLBACK_MESSAGE,
    GREETING_MESSAGE,
    CONVERSATION_CHECK_PROMPT,
    ANSWER_FROM_HISTORY_PROMPT,
)
from src.graph.schema import AgentState
from langchain_core.tools import tool
from src.retrieval.retriever import PolicyRetriever
from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm_model: ChatOpenAI | ChatGoogleGenerativeAI
retriever: PolicyRetriever


def _extract_text(response) -> str:
    content = getattr(response, "content", "")

    if isinstance(content, list):
        if content and isinstance(content[0], dict):
            return str(content[0].get("text", "")).strip()
        return str(content[0]).strip()
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def is_simple_greeting(message: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\s']", " ", message.lower()).strip()
    if not normalized:
        return False

    greeting_terms = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "greetings",
        "how are you",
        "how are you doing",
        "how's it going",
    ]

    words = normalized.split()
    if len(words) > 15:
        return False

    return any(term in normalized for term in greeting_terms)


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

    if is_simple_greeting(last_msg):
        return {"next_node": "greeting"}

    messages = [
                SystemMessage(content=(GUARDRAIL_PROMPT)),
                HumanMessage(content=last_msg),
            ]
    response = llm_model.invoke(messages)

    label = _extract_text(response).strip().upper()

    if "UNSAFE" in label:
        return {"next_node": "unsafe"}
    if "OFF_TOPIC" in label or "OFF-TOPIC" in label:
        return {"next_node": "off_topic"}
    if "SAFE" in label:
        return {"next_node": "safe"}


def routing_after_guardrail(state: AgentState):
    if state["next_node"] == "safe":
        return "conversation_check"
    return state["next_node"]


def handle_greeting(state: AgentState):
    return {"messages": [AIMessage(content=GREETING_MESSAGE)]}


def handle_off_topic(state: AgentState):
    return {"messages": [AIMessage(content=OFF_TOPIC_MESSAGE)]}


def handle_unsafe_message(state:AgentState):
    return {"messages" : [AIMessage(content = UNSAFE_MESSAGE)]}

def conversation_follow_up_check(state: AgentState):
    messages = state["messages"]
    if len(messages) < 2:
        return {"next_node": "retrieve"}

    check_messages = [
        SystemMessage(content=CONVERSATION_CHECK_PROMPT),
        *messages,
    ]
    response = llm_model.invoke(check_messages)
    decision = _extract_text(response).strip().upper()

    if "YES" in decision:
        return {"next_node": "answer_from_history"}
    return {"next_node": "retrieve"}


def route_after_conversation_check(state: AgentState):
    return state["next_node"]


def answer_from_history(state: AgentState):
    history_messages = state["messages"]

    if not history_messages:
        return {"messages": [AIMessage(content=FALLBACK_MESSAGE)]}

    prompt_messages = [
        SystemMessage(content=ANSWER_FROM_HISTORY_PROMPT),
        *history_messages,
    ]
    response = llm_model.invoke(prompt_messages)
    return {"messages": [response]}


def retrieve(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1].content

    # Prior chat_history exists. Therefore need to rephrase the query to be a 
    # stand-alone query based on the entire chat history.
    if len(messages) > 1:
        rephrased_chat_history = messages + [
            SystemMessage(content=(
                "Given the user's chat history and the latest user question about HR Policies for some unnamed organization, "
                "rewrite the question accordingly so it is fully self-contained (resolve pronouns like 'she','it', 'that' etc)."
                "You need not worry about which exact organization. "
                "ONLY RETURN THE RE-WRITTEN STANDALONE QUESTION."
                "DO NOT ANSWER THE QUESTION YOURSELF."
            )),
        ]

        standalone_response = llm_model.invoke(rephrased_chat_history)
        standalone_query = ""

        # This check is required if use Gemini 3.x versions,
        # where the content of AIMessage is a list, rather than a string.

        if isinstance(standalone_response.content, list):
            #print("Inside if block...", optimized_response.content[0])
            standalone_query = standalone_response.content[0]['text'].strip()
        elif isinstance(standalone_response.content, str):
            # print("Inside else block...")
            standalone_query = standalone_response.content.strip()

        # The tool will take care of formatting the documents and return only the context for user_query.
        # context = hr_policy_documents_search.invoke({"query": standalone_query})
        context = hr_policy_documents_search.invoke(input=standalone_query)        

    else:  
        search_query_messages = [   SystemMessage(content=(
                                        "Given the user query, Re-write the query accordingly such that it is "
                                        "best optimized and suitable for vectordatabase search for "
                                        "a HR Policy related query for some unnamed organization. "
                                        "You need not worry about which exact organization. "
                                        "DO NOT ASK ANY FOLLOW-UP QUESTIONS."
                                        "ONLY RETURN THE RE-WRITTEN QUESTION."
                                        "DO NOT ANSWER THE QUESTION YOURSELF."
                                    )),
                                    HumanMessage(content=last_message)
                                ]
        optimized_response = llm_model.invoke(search_query_messages)
        # print(f'-'*100)
        # print(optimized_response)
        # print(f'-'*100)

        optimized_search_query = "" 

        # This check is required if use Gemini 3.x versions,
        # where the content of AIMessage is a list, rather than a string.

        if isinstance(optimized_response.content, list):
            #print("Inside if block...", optimized_response.content[0])
            optimized_search_query = optimized_response.content[0]['text'].strip()
        elif isinstance(optimized_response.content, str):
            # print("Inside else block...")
            optimized_search_query = optimized_response.content.strip()

        # The tool will take care of formatting the documents and return only the context for user_query.
        context = hr_policy_documents_search.invoke(input=optimized_search_query)

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
