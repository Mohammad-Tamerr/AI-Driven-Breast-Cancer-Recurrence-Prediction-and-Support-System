from abc import ABC, abstractmethod
class LLMInterface(ABC):

    @abstractmethod
    def set_generation_model(self, model_id: str): 
        pass # Set the model for text generation (1e.g., GPT-4, PaLM2)


    @abstractmethod
    def set_embedding_model(self, model_id: str, embedding_size: int):
        pass # Set the model for text embedding (e.g., Ada-002, E5-Large)

    @abstractmethod
    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int=None,
                            temperature: float = None):
        pass # Generate text based on the prompt (e.g., completion or chat)

    @abstractmethod
    def embed_text(self,
                text: str,
                document_type: str):
        pass # Embed text for the specified document type 

    @abstractmethod
    def construct_prompt(self,
                        prompt: str,
                        role: str):
        pass # Construct a prompt for the LLM based on the role (e.g., system, user, assistant)
