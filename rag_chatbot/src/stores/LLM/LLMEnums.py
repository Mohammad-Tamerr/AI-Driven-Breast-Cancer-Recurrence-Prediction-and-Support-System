from enum import Enum

class LLMEnums(Enum):
    OPENAI = "OPENAI"
    COHERE = "COHERE"
    GEMINI = "GEMINI"

class OpenAIEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class CoHereEnums(Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "CHATBOT"

    DOCUMENT = "search_document"
    QUERY = "search_query"

class GeminiEnums(Enum):
    USER = "user"
    MODEL = "model"


class DocumentTypeEnum(Enum):
    DOCUMENT = "document"
    QUERY = "query"