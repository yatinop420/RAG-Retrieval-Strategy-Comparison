from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL

# PDF file path
PDF_PATH = "documents/deeplearning.pdf"

# Load PDF
def load_pdf():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    return documents


def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    return chunks

# Create Vector Store
def create_vector_store(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="deep_learning_pdf",
        persist_directory="chroma_db"
    )

    return vector_store

def run_ingestion():
    # Step 1: Load PDF
    documents = load_pdf()
    # Step 2: Split PDF into chunks
    chunks = chunk_documents(documents)
     # Step 3: Create embeddings and store in ChromaDB
    vector_store = create_vector_store(chunks)

    print(f"PDF loaded successfully!")
    print(f"Total pages: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")
    print("Embeddings stored in ChromaDB successfully!")

    return vector_store

if __name__ == "__main__":
    run_ingestion()