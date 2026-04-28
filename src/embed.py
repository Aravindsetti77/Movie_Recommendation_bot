import pandas as pd
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
client=chromadb.PersistentClient(path="./movie_db")
model="all-MiniLM-L6-V2"
emb_fn=embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=model,
    device="cuda")
collection=client.get_or_create_collection(name="TMDB",embedding_function=emb_fn,metadata={"hnsw:space": "cosine"})
df=pd.read_csv("Data/TMDB_clear_data.csv")
def create_embeddings():
    batch_size=512
    rows=len(df)
    for i in range(0,rows,batch_size):
        batch_df=df.iloc[i:i+batch_size]
        collection.add(
            documents=batch_df['dna'].tolist(),
            metadatas=[{"title": row['title'], "genres": row['genres_list']} for _, row in batch_df.iterrows()],
            ids=batch_df['id'].astype(str).tolist()
        )
        if i%512==0:
            print(f"processed{i}/{rows}")
def verify_embedding(id):
    result=collection.get(
        ids=[id],
        include=["embeddings", "metadatas", "documents"]
    )
    print(result['embeddings'][0])
verify_embedding("27205")