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
import json

app = Flask(__name__)

load_dotenv()       
pinecone_api_key = os.getenv("PINECONE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = pinecone_api_key
os.environ["GEMINI_API_KEY"] = gemini_api_key

# ✅ قراءة بيانات المرضى
def load_patients():
    try:
        with open('../data/patients.json', 'r', encoding='utf-8') as f:
            patients_list = json.load(f)
            patients_dict = {patient['patient_id']: patient for patient in patients_list}
            print(f"✅ Loaded {len(patients_dict)} patients")
            return patients_dict
    except FileNotFoundError:
        print("⚠️ patients.json not found!")
        return {}

patients_data = load_patients()

# Load embeddings
embedding = download_embeddings()

index_name = "rafeek-bot-v2"

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embedding,
    index_name=index_name
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.4,
    google_api_key=gemini_api_key
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),  
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
        patient_id = data.get("patient_id", "")
        
        if not user_question:
            return jsonify({"error": "No question provided"}), 400
        
        # ✅ جلب بيانات المريض
        if patient_id and patient_id in patients_data:
            patient = patients_data[patient_id]
            
            patient_context = f"""
معلومات المريض:
- الاسم: {patient.get('name')}
- العمر: {patient.get('age')} سنة
- المرحلة: {patient.get('stage')}
- نوع الورم: {patient.get('tumor_type')}
- العلاج الحالي: {', '.join([t['type'] for t in patient.get('treatments', [])])}
"""
            print(f"👤 Patient: {patient_id} - {patient.get('name')}")
        else:
            patient_context = ""
        
        print(f"❓ Question: {user_question}")
        
        # ✅ دمج معلومات المريض مع السؤال
        full_input = f"{patient_context}\n\nالسؤال: {user_question}" if patient_context else user_question
        
        # Get response from RAG chain
        response = rag_chain.invoke({"input": full_input})
        
        print(f"🤖 Answer: {response['answer'][:150]}...")
        
        return jsonify({
            "answer": response["answer"]
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "عذراً، حدث خطأ في معالجة السؤال"}), 500

@app.route("/test")
def test_page():
    return render_template('test_patient_id.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)