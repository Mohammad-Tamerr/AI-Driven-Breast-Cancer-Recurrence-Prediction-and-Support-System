from ..LLMInterface import LLMInterface
from ..LLMEnums import GeminiEnums
import google.generativeai as genai
import logging

class GeminiProvider(LLMInterface):

    def __init__(self, api_key: str, api_url: str=None,
                    default_input_max_characters: int=1000,
                    default_generation_max_output_tokens: int=1000,
                    default_generation_temperature: float=0.1):
        
        self.api_key = api_key
        self.api_url = api_url  # Not used in Gemini

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        # Only configure Gemini if API key is provided
        if self.api_key and self.api_key.strip():
            genai.configure(api_key=self.api_key)
            self.client = True
        else:
            self.client = None

        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int=None,
                            temperature: float = None):
        
        if not self.client:
            self.logger.error("Gemini client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for Gemini was not set")
            return None
        
        try:
            max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
            temperature = temperature if temperature else self.default_generation_temperature

            model = genai.GenerativeModel(self.generation_model_id)
            
            # Convert chat history to Gemini format
            gemini_history = []
            for msg in chat_history:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    role = "user" if msg['role'] in ['user', 'USER'] else "model"
                    gemini_history.append({
                        "role": role,
                        "parts": [msg['content']]
                    })

            # Start chat with history
            chat = model.start_chat(history=gemini_history)
            
            # Generate response
            response = chat.send_message(
                self.process_text(prompt),
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                )
            )

            if response and response.text:
                return response.text
            else:
                self.logger.error("Error while generating text with Gemini")
                return None

        except Exception as e:
            self.logger.error(f"Error generating text with Gemini: {e}")
            return None

    def embed_text(self, text: str, document_type: str = None):
        
        if not self.client:
            self.logger.error("Gemini client was not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for Gemini was not set")
            return None
        
        try:
            result = genai.embed_content(
                model=self.embedding_model_id,
                content=text,
                task_type="retrieval_document" if document_type == "document" else "retrieval_query"
            )

            if result and 'embedding' in result:
                return result['embedding']
            else:
                self.logger.error("Error while embedding text with Gemini")
                return None

        except Exception as e:
            self.logger.error(f"Error embedding text with Gemini: {e}")
            return None

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": self.process_text(prompt)
        }
