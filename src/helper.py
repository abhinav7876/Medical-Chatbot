from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk

def download_embeddings():
    """
    Download and return the HuggingFace embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings

def rerank_docs(query, docs, reranker,top_k=3):
    """Return top_k most relevant documents based on cross-encoder scores."""
    # Create (query, document) pairs
    pairs = [(query, doc.page_content) for doc in docs]
    
    # Get similarity scores
    scores = reranker.predict(pairs)
    
    # Sort docs by score (highest first)
    sorted_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
    
    return sorted_docs[:top_k]


def hyde_query_expansion(query,llm):
    """Generate a hypothetical answer to enrich retrieval."""
    prompt = f"Write a short, factual paragraph that could answer: '{query}'"
    pseudo_answer = llm.invoke(prompt).content
    return pseudo_answer


