from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()       
pinecone_api_key = os.getenv("PINECONE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = pinecone_api_key
os.environ["GEMINI_API_KEY"] = gemini_api_key

# Define paths
data_path = "/Users/belalmohsen/Breast-cancer/data"  # ✅ غيّر المسار النسبي
index_name = "rafeek-bot"

# Load and process documents
print("📄 Loading PDF files...")
extracted_data = load_pdf_files(data_path)
print(f"✅ Loaded {len(extracted_data)} pages")

minimal_docs = filter_to_minimal_docs(extracted_data)
texts_splits = text_split(minimal_docs)
print(f"✅ Split into {len(texts_splits)} chunks")

# Download embeddings
print("🔽 Loading embeddings model...")
embedding = download_embeddings()
print("✅ Embeddings ready")

# Initialize Pinecone
print("🔌 Connecting to Pinecone...")
pc = Pinecone(api_key=pinecone_api_key)

# Create index if not exists
if not pc.has_index(index_name):
    print(f"🆕 Creating index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"✅ Index '{index_name}' exists")

# Store documents
print("📤 Storing documents in Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=texts_splits,
    embedding=embedding,
    index_name=index_name
)
print(f"🎉 Success! Stored {len(texts_splits)} chunks!")