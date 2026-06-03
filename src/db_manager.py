import chromadb
import torch
import pandas as pd
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
class DBManager:
    _client=None
    def __init__(self, path="./movie_db"):
        self._client = chromadb.PersistentClient(path=path)
        self.emb_fn = SentenceTransformerEmbeddingFunction(
            model_name="multi-qa-mpnet-base-dot-v1",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    def get_collection(self,name):
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=self.emb_fn,
            metadata={"hnsw:space": "cosine"}
        )
    def delete_collection(self, name):
        try:
            self._client.delete_collection(name=name)
            print(f"Collection '{name}' deleted.")
        except Exception as e:
            print(f"Error deleting collection: {e}")
    def delete_all(self):
        collections = self._client.list_collections()

        for col in collections:
            print(f"Deleting: {col.name}")
            self._client.delete_collection(name=col.name)

        print("All collections deleted.")
if __name__=="__main__":
    db=DBManager()
    db.delete_all()