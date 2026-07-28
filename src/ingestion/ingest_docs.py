from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from uuid import uuid4
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings
from src.retrieval.retriever import PolicyRetriever


def _load_files(files: list[Path]) -> list[Document]:
    pages : list[Document] = []
    for file in files:
        pdf_loader = PyPDFLoader(file_path=file)
        for page in pdf_loader.load():

            # page content processing.
            replacement = re.findall(r"\d+ \n \n", page.page_content)[0]
            
            # Metadata processing
            new_metadata = {}
            new_metadata['source'] = file.name
            new_metadata['total_pages'] = page.metadata.get('total_pages')
            new_metadata['creation_time'] = page.metadata.get('creationdate')
            new_metadata['modified_time'] = page.metadata.get('moddate')

            if isinstance(page.metadata.get("page"), int):
                new_metadata['page'] = page.metadata["page"] + 1

            new_doc = Document(page_content = page.page_content.replace(replacement, ""),
                       metadata = new_metadata)
            
            pages.append(new_doc)

    return pages


def _split_documents(docs: list[Document], chunk_size, chunk_overlap) -> list[Document]:

    # used \n \n as a separator, since paragraph splitting were coming like this instead of \n\n
    splitter = RecursiveCharacterTextSplitter(separators=["SECTION","\n \n"],
                                          chunk_size = chunk_size,
                                          chunk_overlap = chunk_overlap,
                                          # The index (the exact character position (0-indexed) where the chunk begins in the original text)
                                          #  is stored inside the metadata dictionary of each generated Document object under the key "start_index"
                                          add_start_index=True,  
                                          )
    splitted_documents : list[Document] = splitter.split_documents(docs)
    return splitted_documents

def ingest_data_directory(settings: Settings, retriever: PolicyRetriever) -> tuple[int, int, list[str]]:
    pdf_files = sorted(settings.data_dir.glob("*.pdf"))
    if not pdf_files:
        return 0,0, []

    loaded_documents = _load_files(files=pdf_files)
    splitted_documents = _split_documents(docs=loaded_documents,
                                        chunk_size=settings.chunk_size,
                                        chunk_overlap= settings.chunk_overlap)

    

    document_ids = [str(uuid4()) for _ in range(len(splitted_documents))]

    retriever.store_documents(splitted_documents, document_ids)
    
    return len(pdf_files), len(splitted_documents), document_ids


