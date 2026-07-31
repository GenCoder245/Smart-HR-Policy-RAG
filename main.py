from config import Settings, get_settings
from src.retrieval.retriever import PolicyRetriever
from src.ingestion.ingest_docs import ingest_data_directory
from src.models.llm_models import get_language_model
import structlog
from my_logging import configure_logging

# Import below Only for testing
from src.prompt.prompts import SYSTEM_PROMPT 
from langchain_core.prompts import ChatPromptTemplate

logger = structlog.get_logger()

policy_settings : Settings = get_settings()
configure_logging(policy_settings.log_level)

policy_retriever = PolicyRetriever(settings=policy_settings)
policy_retriever.initialize()

if policy_settings.re_ingest_docs:
    logger.info(f"Re-ingesting documents into the vectordb.")
    no_of_files, no_of_chunks, chunk_ids = ingest_data_directory(settings=policy_settings, retriever= policy_retriever)
    logger.info(f"Inserted {no_of_chunks} no.of chunks into the vectordb.")

llm_model = get_language_model(llm_settings=policy_settings)

# Just for checking:
user_query = "what is the performance management process followed here ?"
# user_query = "how many sick leaves are allowed in this organization ?"
# user_query = "tell me about the work from home guidelines ?"

retrieved_documents = policy_retriever.retrieve_documents(user_query)
# Can also use below code
#retrieved_documents = policy_retriever.search_with_score(user_query)

formatted_docs = policy_retriever.format_documents(retrieved_documents)
#logger.info(f"retrieved documents: {retrieved_documents}")
logger.info(f"retrieved relevant documents...")

template = ChatPromptTemplate([
    ("system",SYSTEM_PROMPT),
    ("human",user_query),
])
llm_input_messages = template.format_messages(context = formatted_docs)

logger.info(f"The context is sent to the LLM...")

ai_response = llm_model.invoke(llm_input_messages)

logger.info(f"The AI Response: {ai_response.content}")


