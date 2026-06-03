import pandas as pd
import numpy as np
df=pd.read_csv("Data/Merge.csv")
# Vectorized operations for better performance with natural language structure
df["dna"] = (
    "Title: " + df["original_title"].astype(str) + ". " +
    "Genres: " + df["genres"].astype(str) + ". " +
    "Plot overview: " + df["overview"].astype(str)
)
cols=['id', 'original_title', 'genres', 'dna']
final_df=df[cols].dropna(subset="dna").drop_duplicates(subset="id")
final_df.to_csv("Data/TMDB_clear_data.csv",index=False)

print(f"{len(df)} to {len(final_df)}")