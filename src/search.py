import pandas as pd
import chromadb
import torch
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from embed import Embeddings
from db_manager import DBManager
class Recommender:
    def __init__(self):
        self.db = DBManager()
        self.collection = self.db.get_collection(name="TMDB")
    def get_recommendations(self, movie_title, n_results=10):

        match_results = self.collection.query(
            query_texts=[movie_title],
            n_results=1,
            include=["embeddings", "metadatas"]
        )

        if not match_results['ids'][0]:
            return {"error": "Movie not found"}

        target_movie_id = match_results['ids'][0][0]
        target_title = match_results['metadatas'][0][0].get('title', 'Unknown')
        target_embedding = match_results['embeddings'][0][0]

        similarity_results = self.collection.query(
            query_embeddings=[target_embedding],
            n_results=n_results + 1,
            include=["metadatas"]
        )

        final_ids = []
        final_metadatas = []
        for i, m_id in enumerate(similarity_results['ids'][0]):
            if m_id != target_movie_id and len(final_ids) < n_results:
                final_ids.append(m_id)
                final_metadatas.append(similarity_results['metadatas'][0][i])

        return {
            "matched_title": target_title,
            "ids": final_ids,
            "metadatas": final_metadatas
        }

if __name__ == "__main__":
    rec = Recommender()
    results = rec.get_recommendations("Inception")
    print(results)