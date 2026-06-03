import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from dotenv import load_dotenv
import sys

# Ensure src is in path so we can import search
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from search import Recommender

load_dotenv(override=True)
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

app = FastAPI(title="Movie Recommender")

# Enable CORS for external integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins; change this to your friend's domain in production!
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# Initialize Recommender
recommender = Recommender()

class RecommendationRequest(BaseModel):
    title: str



@app.post("/api/recommend")
async def get_recommendations(req: RecommendationRequest):
    try:
        # Get recommendations from local DB
        results = recommender.get_recommendations(req.title, n_results=10)
        
        if "error" in results:
            raise HTTPException(status_code=404, detail=results["error"])
            
        movie_ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])
        
        # If no results, return empty
        if not movie_ids:
            return {"recommendations": []}

        # Combine results
        recommendations = []
        for i in range(len(movie_ids)):
            recommendations.append({
                "id": movie_ids[i],
                "title": metadatas[i].get("title", "Unknown Title")
            })

        return {
            "matched_title": results.get("matched_title"),
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Bound to 0.0.0.0 to allow external access (vs 127.0.0.1 for local only)
    uvicorn.run(app, host="0.0.0.0", port=8000)
