import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# List all available models
print("🤖 Available Gemini Models:\n")
print("="*60)

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"✅ {model.name}")
        print(f"   Description: {model.description}")
        print(f"   Input limit: {model.input_token_limit}")
        print(f"   Output limit: {model.output_token_limit}")
        print("-"*60)