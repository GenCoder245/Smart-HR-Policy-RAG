from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_core.vectorstores import VectorStoreRetriever

from config import Settings, get_settings

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


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

    # Check the usage of this function
    def search_with_score(self, query: str):
        store = self._require_store()
        candidates = store.similarity_search_with_score(
            query,
            k=self.settings.retrieval_limit,
        )
        if not candidates:
            return []

        return candidates

    def search(self, query: str) -> list[Document]:
        store = self._require_store()
        candidates = store.similarity_search(
            query,
            k=self.settings.retrieval_limit,
        )
        if not candidates:
            return []
    
        return candidates

    def store_documents(self,documents: list[Document], ids: list[str]):
        store = self._require_store()
        store.add_documents(documents=documents, ids=ids)


    def format_documents(self, documents: list[Document]):
        formatted_contents = "\n\n".join([doc.page_content for doc in documents])
        return formatted_contents

    def retrieve_documents(self, query: str) -> list[Document]:
        retriever = self._require_retriever()

        candidates = retriever.invoke(query)
        if not candidates:
            return []
    
        return candidates
    

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











