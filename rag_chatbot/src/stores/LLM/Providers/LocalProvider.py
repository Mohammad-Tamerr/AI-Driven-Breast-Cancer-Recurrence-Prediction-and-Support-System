from ..LLMInterface import LLMInterface
import hashlib
import logging

class LocalProvider(LLMInterface):
    """A small deterministic local embedding provider used for testing/indexing without external APIs.
    It produces an embedding vector of fixed length derived from a SHA256 hash of the text.
    """

    def __init__(self, api_key: str = None, default_input_max_characters: int=1000,
                 default_generation_max_output_tokens: int=1000, default_generation_temperature: float=0.1):
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.embedding_model_id = None
        self.embedding_size = None

        self.logger = logging.getLogger(__name__)

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def set_generation_model(self, model_id: str):
        # Not used for local provider
        self.generation_model_id = model_id

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def embed_text(self, text: str, document_type: str = None):
        if not self.embedding_size:
            self.logger.error("Embedding model/size is not set for LocalProvider")
            return None

        t = self.process_text(text)

        # Use SHA256 digest to derive deterministic bytes
        digest = hashlib.sha256(t.encode('utf-8')).hexdigest()

        # Convert hex digest into a vector of numbers between -1 and 1
        vec = []
        # use pairs of hex chars to create bytes, cycle digest if needed
        i = 0
        while len(vec) < self.embedding_size:
            pair = digest[(i*2) % len(digest): (i*2 + 2) % len(digest)]
            if not pair:
                pair = digest[0:2]
            val = int(pair, 16)
            # map 0..255 to -1..1
            norm = (val / 127.5) - 1.0
            vec.append(norm)
            i += 1

        return vec

    def generate_text(self, prompt: str, chat_history: list = [], max_output_tokens: int=None, temperature: float=None):
        # Simple stub for generation: return the prompt truncated
        return self.process_text(prompt)

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "content": self.process_text(prompt)}
