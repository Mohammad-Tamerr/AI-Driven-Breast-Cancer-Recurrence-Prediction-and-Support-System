from ..VectorDBInterface import VectorDBInterface
from typing import List
import math
import logging

class InMemoryDBProvider(VectorDBInterface):
    def __init__(self, db_path: str = None, distance_method: str = "cosine"):
        self.store = {}  # collection_name -> { "vectors": [vec], "texts": [text], "metadata": [meta], "ids": [id] }
        self.distance_method = distance_method
        self.logger = logging.getLogger(__name__)

    def connect(self):
        # no-op for in-memory
        return True

    def disconnect(self):
        self.store = {}

    def is_collection_existed(self, collection_name: str) -> bool:
        return collection_name in self.store

    def list_all_collections(self) -> List:
        return list(self.store.keys())

    def get_collection_info(self, collection_name: str) -> dict:
        if not self.is_collection_existed(collection_name):
            return {}
        col = self.store[collection_name]
        return {"size": len(col["vectors"]) }

    def delete_collection(self, collection_name: str):
        if self.is_collection_existed(collection_name):
            del self.store[collection_name]

    def create_collection(self, collection_name: str, embedding_size: int, do_reset: bool = False):
        if do_reset and self.is_collection_existed(collection_name):
            self.delete_collection(collection_name)

        if not self.is_collection_existed(collection_name):
            self.store[collection_name] = {"vectors": [], "texts": [], "metadata": [], "ids": []}
            return True
        return False

    def insert_one(self, collection_name: str, text: str, vector: list, metadata: dict = None, record_id: str = None):
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Can not insert new record to non-existed collection: {collection_name}")
            return False
        col = self.store[collection_name]
        col["vectors"].append(vector)
        col["texts"].append(text)
        col["metadata"].append(metadata)
        col["ids"].append(record_id)
        return True

    def insert_many(self, collection_name: str, texts: list, vectors: list, metadata: list = None, record_ids: list = None, batch_size: int = 50):
        if metadata is None:
            metadata = [None] * len(texts)
        if record_ids is None:
            record_ids = [None] * len(texts)

        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Collection does not exist: {collection_name}")
            return False

        col = self.store[collection_name]
        for i in range(len(texts)):
            col["vectors"].append(vectors[i])
            col["texts"].append(texts[i])
            col["metadata"].append(metadata[i])
            col["ids"].append(record_ids[i])

        return True

    def _cosine_similarity(self, a, b):
        # safe cosine similarity
        dot = sum(x*y for x,y in zip(a,b))
        norm_a = math.sqrt(sum(x*x for x in a))
        norm_b = math.sqrt(sum(x*x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search_by_vector(self, collection_name: str, vector: list, limit: int = 5):
        if not self.is_collection_existed(collection_name):
            return []
        col = self.store[collection_name]
        scores = []
        for i, v in enumerate(col["vectors"]):
            score = self._cosine_similarity(vector, v)
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        out = []
        for idx, score in scores[:limit]:
            out.append({"payload": {"text": col["texts"][idx], "metadata": col["metadata"][idx]}, "score": score})
        return out
