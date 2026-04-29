import pandas as pd
import ast
import json
def extract_movies(col):
    try:
        items=ast.literal_eval(col) if isinstance(col,str) else col
        return "".join([i['name'] for i in items]) if items else ""
    except (ValueError, SyntaxError, TypeError):
        return ""
df=pd.read_csv("Data/TMDB_movie_dataset_v11.csv")
cleaned_df=df[
    (df["runtime"]>=60) &
    (df["status"]=="Released")
    ].copy()
cleaned_df['keywords_list'] = cleaned_df['keywords'].apply(extract_movies)
def build_dna(row):
    parts = [
        str(row['title']),
        (str(row['genres']))*3,
        (str(row['keywords_list'])),
        str(row['overview']),
        str(row["popularity"])*2
    ]
    return "".join(parts).lower()
cleaned_df["dna"]=cleaned_df.apply(build_dna,axis=1)
cols=['id', 'title', 'genres_list', 'dna']
final_df=cleaned_df[cols].dropna(subset="dna")
final_df.to_csv("Data/TMDB_clear_data.csv",index=False)
print(f"{len(df)} to {len(final_df)}")