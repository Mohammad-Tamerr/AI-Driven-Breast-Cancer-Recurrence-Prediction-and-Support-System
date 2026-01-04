from enum import Enum

class VectorDBEnums(Enum):
    QDRANT = "QDRANT"
    INMEMORY = "INMEMORY"

class DistanceMethodEnums(Enum):
    COSINE = "cosine"
    DOT = "dot"