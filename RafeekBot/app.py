from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()       
pinecone_api_key = os.getenv("PINECONE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = pinecone_api_key
os.environ["GEMINI_API_KEY"] = gemini_api_key

# Load embeddings
embedding = download_embeddings()

index_name = "rafeek-bot-v2"

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embedding,
    index_name=index_name
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # ✅ زودها من 3 لـ 5 عشان نتائج أكتر
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  # ✅ غيّر من gemini-1.5-flash
    temperature=0.4,  # ✅ خليها 0.4 للتوازن بين الدقة والطبيعية
    google_api_key=gemini_api_key
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),  # 
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/chat", methods=["POST"])  
def chat():
    try:
        data = request.json 
        user_question = data.get("question", "")
        
        if not user_question:
            return jsonify({"error": "No question provided"}), 400
        
        print(f"User question: {user_question}")
        
        # Get response from RAG chain
        response = rag_chain.invoke({"input": user_question})
        
        print(f"Bot response: {response['answer']}")
        
        return jsonify({
            "answer": response["answer"]
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "عذراً، حدث خطأ في معالجة السؤال"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)