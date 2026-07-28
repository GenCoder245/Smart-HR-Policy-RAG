from config import Settings, get_settings
from src.retrieval.retriever import PolicyRetriever
from src.ingestion.ingest_docs import ingest_data_directory
import structlog
from my_logging import configure_logging


logger = structlog.get_logger()

policy_settings : Settings = get_settings()
configure_logging(policy_settings.log_level)

policy_retriever = PolicyRetriever(settings=get_settings())
policy_retriever = policy_retriever.initialize()

no_of_files, no_of_chunks, chunk_ids = ingest_data_directory(settings=policy_settings, retriever= policy_retriever)

logger.info(f"Inserted {no_of_chunks} no.of chunks into the vectordb.")




