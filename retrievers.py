
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL
#from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
#hybridsearching
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
#from langchain.retrievers import EnsembleRetriever
from langchain_classic.retrievers import EnsembleRetriever
import re
# Load Existing Vector Database
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = Chroma(
        collection_name="deep_learning_pdf",
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    return vector_store


# Similarity Search
def similarity_search(query, k=4):
    vector_store = load_vector_store()

    results = vector_store.similarity_search(
        query,
        k=k
    )

    return results


# MMR (Maximum Marginal Relevance) Search

def mmr_search(query, k=4, fetch_k=20, lambda_mult=0.5):

    vector_store = load_vector_store()

    results = vector_store.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult
    )

    return results

# if __name__ == "__main__":

#     query = "What is backpropagation?"

#     results = mmr_search(query, k=4)

#     for i, doc in enumerate(results, 1):
#         print(f"\nResult {i}:")
#         print(doc.page_content)
#         print("Metadata:", doc.metadata)

# Multi-Query Search


# Multi-Query Search
def multi_query_search(query, k=4):
    vector_store = load_vector_store()

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(
            search_kwargs={"k": k}
        ),
        llm=llm
    )

    results = retriever.invoke(query)
    return results




# BM25 (Keyword-Based) Search

# Tokenizer function
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# BM25 Search
def bm25_search(query, k=4):

    vector_store = load_vector_store()

    data = vector_store.get(
        include=["documents", "metadatas"]
    )

    documents = [
        Document(
            page_content=content,
            metadata=metadata or {}
        )
        for content, metadata in zip(
            data["documents"],
            data["metadatas"]
        )
        if content
    ]

    # Create BM25 Retriever
    bm25_retriever = BM25Retriever.from_documents(
        documents,
        preprocess_func=tokenize
    )

    bm25_retriever.k = k

    # Retrieve relevant documents
    results = bm25_retriever.invoke(query)

    # Display results
    for i, doc in enumerate(results, 1):
        print(f"\nRank {i}")
        print("Page:", doc.metadata.get("page_label"))
        print("Content:", doc.page_content[:500])
        print("-" * 60)

    return results

# Hybrid Search (Vector + BM25)
def hybrid_search(query, k=4, alpha=0.5):

    vector_store = load_vector_store()

    # Get documents from ChromaDB
    data = vector_store.get(
        include=["documents", "metadatas"]
    )

    # Convert into LangChain Documents
    documents = [
        Document(
            page_content=content,
            metadata=metadata or {}
        )
        for content, metadata in zip(
            data["documents"],
            data["metadatas"]
        )
        if content
    ]

    # BM25 Retriever
    bm25_retriever = BM25Retriever.from_documents(
    documents,
        preprocess_func=tokenize
    )
    bm25_retriever.k = k

    # Dense Vector Retriever
    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )

    # Combine both retrievers
    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            vector_retriever,
            bm25_retriever
        ],
        weights=[
            alpha,
            1 - alpha
        ]
    )

    # Execute Hybrid Search
    results = ensemble_retriever.invoke(query)

    return results



if __name__ == "__main__":
    query = "What is backpropagation?"

    results = hybrid_search(query, k=4)

    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print("Page:", doc.metadata.get("page_label"))
        print("Content:", doc.page_content[:500])
        print("-" * 60)