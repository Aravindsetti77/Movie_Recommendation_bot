import pandas as pd
import numpy as np
import chromadb
import torch
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
class Embeddings:
    def __init__(self,client):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer('All-MiniLM-L6-v2', device=self.device)
        self.emb_fn=SentenceTransformerEmbeddingFunction(model_name="all_MiniLM-L6-V2",device="cuda")
        self.collectioncollection=client.get_or_create_collection(name="TMDB",embedding_function=self.emb_fn,metadata={"hnsw:space": "cosine"})
        self.df=pd.read_csv("Data/TMDB_clear_data.csv")
    def create_embeddings(self):
        batch_size=512
        rows=len(self.df)
        for i in range(0,rows,batch_size):
            batch_df=self.df.iloc[i:i+batch_size]
            self.collection.add(
                documents=batch_df['dna'].tolist(),
                metadatas=[{"title": row['title'], "genres": row['genres_list']} for _, row in batch_df.iterrows()],
                ids=batch_df['id'].astype(str).tolist()
            )
            if i%512==0:
                print(f"processed{i}/{rows}")
    def verify_embedding(self,id):
        result=self.collection.get(
            ids=[str(id)],
            include=["embeddings", "metadatas", "documents"]
        )
        print(result['embeddings'][0])