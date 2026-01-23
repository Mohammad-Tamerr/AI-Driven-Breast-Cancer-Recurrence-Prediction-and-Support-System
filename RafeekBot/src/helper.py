from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document
import os

def load_pdf_files(data_path: str) -> List[Document]:
    """Load all PDF files from a directory."""
    documents = []
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(data_path) if f.endswith('.pdf')]
    
    print(f"📚 Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        file_path = os.path.join(data_path, pdf_file)
        try:
            print(f"📄 Loading: {pdf_file}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"✅ Loaded {len(docs)} pages from {pdf_file}")
        except Exception as e:
            print(f"⚠️ Error loading {pdf_file}: {str(e)}")
            print(f"⏭️ Skipping this file...")
            continue
    
    print(f"\n✅ Total documents loaded: {len(documents)}")
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Keep only 'source' in metadata."""
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

def text_split(minimal_docs: List[Document]) -> List[Document]:
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk

def download_embeddings():
    """Download multilingual HuggingFace embeddings model."""
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings