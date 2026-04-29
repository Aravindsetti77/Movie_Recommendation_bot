import pandas as pd
import numpy as np
import chromadb
import torch
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
from db_manager import DBManager
class Embeddings:
    def __init__(self):
        self.db = DBManager()
        self.collection = self.db.get_collection("TMDB")
    def create_embeddings(self, csv_path="Data/TMDB_clear_data.csv"):
        df = pd.read_csv(csv_path)
        df['id'] = df['id'].astype(str)
        df['dna'] = df['dna'].fillna("")
        batch_size = 512
        rows = len(df)
        for i in range(0, rows, batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            self.collection.upsert(
                documents=batch_df['dna'].tolist(),
                metadatas=[
                    {
                        "title": row['title'], 
                        "genres": str(row['genres_list']) # Chroma prefers simple types in metadata
                    } for _, row in batch_df.iterrows()
                ],
                ids=batch_df['id'].tolist()
            )
            print(f"Processed {min(i + batch_size, rows)}/{rows}")
if __name__=="__main__":
    embedder=Embeddings()
    embedder.create_embeddings()