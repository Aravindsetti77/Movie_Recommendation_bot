import pandas as pd
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
client=chromadb.PersistentClient(path="./movie_db")
model="all-MiniLM-L6-V2"
emb_fn=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)
collection=client.get_or_create_collection(name="TMDB",embedding_function=emb_fn,metadata={"hnsw:space": "cosine"})
df=pd.read_csv("Data/TMDB_clear_data.csv")
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