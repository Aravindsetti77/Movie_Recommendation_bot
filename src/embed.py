import pandas as pd
import numpy as np
import chromadb
import torch
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
from db_manager import DBManager
from tqdm import tqdm
class Embeddings:
    def __init__(self):
        self.db = DBManager()
    def create_embeddings(self, csv_path="Data/TMDB_clear_data.csv"):
        self.db.delete_collection("TMDB")
        self.collection = self.db.get_collection(name="TMDB")
        df = pd.read_csv(csv_path)
        df['id'] = df['id'].astype(str)
        df['dna'] = df['dna'].fillna("")
        
        # Load model directly to control batch size for GPU utilization
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        model = SentenceTransformer("multi-qa-mpnet-base-dot-v1", device=device)
        
        batch_size = 5000
        rows = len(df)
        for i in tqdm(range(0, rows, batch_size), desc="Embedding Movies"):
            batch_df = df.iloc[i:i+batch_size]
            docs = batch_df['dna'].tolist()
            
            # Encode with a large batch size to saturate the GPU
            embeddings = model.encode(docs, batch_size=512, show_progress_bar=False).tolist()
            
            self.collection.upsert(
                embeddings=embeddings,
                documents=docs,
                metadatas=[
                    {
                        "title": row['original_title'],
                    } for _, row in batch_df.iterrows()
                ],
                ids=batch_df['id'].tolist()
            )
if __name__=="__main__":
    embedder=Embeddings()
    embedder.create_embeddings()