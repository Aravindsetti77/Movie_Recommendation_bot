import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from search import Recommender



app = FastAPI(title="Movie Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = Recommender()

class RecommendationRequest(BaseModel):
    title: str

@app.post("/api/recommend")
async def get_recommendations(req: RecommendationRequest):
    try:

        results = recommender.get_recommendations(req.title, n_results=10)

        if "error" in results:
            raise HTTPException(status_code=404, detail=results["error"])

        movie_ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])

        if not movie_ids:
            return {"recommendations": []}

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

    uvicorn.run(app, host="0.0.0.0", port=8000)
