from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_core.vectorstores import VectorStoreRetriever

from config import Settings, get_settings

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langsmith import traceable

from src.retrieval.reranker import CohereReranker

class DenseEmbeddings:
    def __init__(self, primary, fallback) -> None:
        self.primary = primary
        self.fallback = fallback

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self.primary.embed_documents(texts)
        except Exception:
            return self.fallback.embed_documents(texts)

    
    def embed_query(self, text: str) -> list[float]:
        try:
            return self.primary.embed_query(text)
        except Exception:
            return self.fallback.embed_query(text)


class PolicyRetriever:
    def __init__(self, settings:Settings):
        self.settings = settings
        self.collection_name = settings.qdrant_collection
        self.qdrant_url = settings.qdrant_url
        self.qdrant_api_key = settings.qdrant_api_key_value
        self.vector_dimension = settings.embedding_dimension

        if self.qdrant_api_key:
            self.client = QdrantClient(url=self.qdrant_url,
                                          api_key= self.qdrant_api_key)
        else:
            self.client = QdrantClient(url=self.qdrant_url)

        self.dense_embeddings = DenseEmbeddings(
            primary=OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                dimensions = settings.embedding_dimension,
                api_key=settings.openai_api_key,
            ),
            fallback=GoogleGenerativeAIEmbeddings(
                model=settings.gemini_embedding_model,
                output_dimensionality = settings.embedding_dimension,
                api_key=settings.google_api_key,
            ),
        )

        self.sparse_embeddings = FastEmbedSparse(model_name=settings.sparse_embedding_model,
                                cache_dir=settings.cache_dir)

        self.cohere_reranker: CohereReranker | None = None

        if settings.cohere_api_key:
            self.cohere_reranker = CohereReranker(
                                api_key=settings.cohere_api_key, 
                                model=self.settings.cohere_rerank_model,  
                                top_k=self.settings.rerank_limit,
                            )                  


        


    def initialize(self):
        collections = self.client.get_collections()
        collection_names = [collection.name for collection in collections.collections]

        if self.collection_name not in collection_names:
            print(f"Collection {self.collection_name} doesnt exist. Creating the collection...")
            collection_created = self.client.create_collection(collection_name=self.collection_name,
                                        vectors_config = {
                                            "dense": VectorParams(
                                                size = self.vector_dimension,
                                                distance= Distance.COSINE,
                                            )
                                        },

                                        sparse_vectors_config= {
                                            "sparse" : SparseVectorParams(
                                                index = SparseIndexParams(on_disk = False)
                                            )
                                        },

                                                            )

            if collection_created:
                print(f"Collection {self.collection_name} created successfully...")
        else:
            print(f"Collection {self.collection_name} exists...")

        self._store = QdrantVectorStore(client=self.client,
                                        collection_name = self.collection_name,
                                        embedding= self.dense_embeddings.primary, # Check this
                                        retrieval_mode = RetrievalMode.HYBRID,
                                        distance = Distance.COSINE,
                                        sparse_embedding = self.sparse_embeddings,
                                        vector_name = "dense",
                                        sparse_vector_name = "sparse",
                                        )

        self._retriever = self._store.as_retriever(search_type = 'similarity',
                                                   search_kwargs={"k": self.settings.retrieval_limit})

        # return self

    # For future use
    def search_with_score(self, query: str):
        store = self._require_store()
        candidates = store.similarity_search_with_score(
            query,
            k=self.settings.retrieval_limit,
        )
        if not candidates:
            return []

        return candidates

    # For future use
    def search(self, query: str) -> list[Document]:
        store = self._require_store()
        candidates = store.similarity_search(
            query,
            k=self.settings.retrieval_limit,
        )
        if not candidates:
            return []
    
        return candidates

    
    @traceable(run_type="embedding", name="store embeddings")
    def store_documents(self,documents: list[Document], ids: list[str]):
        store = self._require_store()
        store.add_documents(documents=documents, ids=ids)


    # tool, chain, llm, retriever, embedding, prompt, parser are the only valid values for "run_type" in the @traceable decorator. 
    # run_type="parser"-> because it is for parsing and formatting the retrieved documents into a suitable format to pass as context to the LLM. 
    @traceable(run_type="parser", name="format documents")
    def format_documents(self, documents: list[Document]):
        formatted_contents = "\n\n".join([doc.page_content for doc in documents])
        return formatted_contents


    @traceable(run_type="retriever", name="retrieve relevant context")
    def retrieve_documents(self, query: str) -> list[Document]:
        retriever = self._require_retriever()

        candidates = retriever.invoke(query)
        if not candidates:
            return []
    
        return candidates


    @traceable(run_type="retriever", name="re-rank documents(cohere)")
    def rerank_documents(self, query: str, ranked_docs: list[Document]) -> list[Document]:
        reranked_docs = None

        if not ranked_docs:
            return []
        
        # apply cohere re-ranking
        if self.cohere_reranker:
            try:
                reranked_docs = self.cohere_reranker.rerank_docs(query, ranked_docs)
            except Exception as e:
                print(f"Cohere re-ranker failed due to : {e}") 
        
        if not reranked_docs:
            return []
        return reranked_docs
    

    def _require_store(self) -> QdrantVectorStore:
        if self._store is None:
            raise RuntimeError("PolicyRetriever.initialize() must be called before use")
        return self._store

    def _require_retriever(self) -> VectorStoreRetriever:
        if self._retriever is None:
            raise RuntimeError("PolicyRetriever.initialize() must be called before use")
        return self._retriever

    def close(self) -> None:
        self.client.close()
        


if __name__ == "__main__":
    policy_retriever = PolicyRetriever(settings=get_settings())
    policy_retriever.initialize()











