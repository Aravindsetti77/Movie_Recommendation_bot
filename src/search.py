import pandas as pd
import chromadb
import torch
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from embed import Embeddings
from db_manager import DBManager
class Recommender:
    def __init__(self):
        self._client=chromadb.PersistentClient(path="./movie_db")
        self.collection=self._client.get_or_create_collection(name="TMDB")
    def get_recommendations(self,movie_title, n_results=50):
        # movie_data = self.collection.get(
        #     where={"title": movie_title},
        #     include=["documents"]
        # )
        # movie_dna=movie_data["documents"][0][0]
        
        results = self.collection.query(
            query_texts=[movie_title],
            n_results=n_results,
            include=["embeddings","metadatas"]
        )
        return results['metadatas'][0]
rec=Recommender()
results=rec.get_recommendations("Interstellar")
print(results)