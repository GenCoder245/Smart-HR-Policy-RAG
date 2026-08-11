from typing import List, Optional
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from pydantic import SecretStr


class CohereReranker:
    """
    LangChain-based Cohere re-ranker wrapper.
    It calls Cohere's `rerank` endpoint and returns the documents reordered by score.

    """

    def __init__(self, api_key: SecretStr ,model: Optional[str] = None, top_k: int = 5) :
        if not api_key:
            raise ValueError("Cohere API key is required for reranking")

        self.api_key = api_key
        self.model = model or "rerank-english-v3.0"
        self.top_k = top_k

        self.reranker = CohereRerank(
            cohere_api_key=api_key,
            model="rerank-english-v3.0",
            top_n=top_k
        )
    

    def rerank_docs(self, query: str, documents: List[Document]) -> List[Document]:
        """Return documents re-ordered by relevance to `query`.

        Keeps only the top `top_k` documents (or fewer if fewer inputs).
        """
        if not documents:  # If Ranked-docuemnts is empty, return empty list
            return []

        doc_texts = [doc.page_content for doc in documents]

        response = self.reranker.rerank(model=self.model, query=query, documents=doc_texts)

        raw_results = getattr(response, "results", response)

        scored: List[tuple[int, float]] = []
        for r in raw_results:
            idx = getattr(r, "index", None) if not isinstance(r, dict) else r.get("index")
            if isinstance(r, dict):
                score = r.get("score") or r.get("relevance_score")
            else:
                score = getattr(r, "score", None) or getattr(r, "relevance_score", None)

            if idx is None:
                continue
            scored.append((int(idx), float(score) if score is not None else 0.0))

        scored_sorted = sorted(scored, key=lambda t: t[1], reverse=True)
        # top_indices = [idx for idx, _ in scored_sorted][: min(self.top_k, len(documents))]

        if scored_sorted:
            return [documents[idx] for idx, _ in scored_sorted]
        else:
            return []