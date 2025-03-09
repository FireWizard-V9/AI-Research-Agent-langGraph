from fastapi import FastAPI
import asyncio
from supervisor import run_supervisor_flow
from fastapi.middleware.cors import CORSMiddleware

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Allow Streamlit to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/search/")
async def search(query: str):
    """Runs research query and returns structured results."""
    print(f"🔍 Searching for '{query}'...")

    # ✅ Run the research supervisor
    results = await run_supervisor_flow(query)

    return {
        "final_response": results["final_response"],
        "raw_results": results["raw_results"]
    }

# ✅ Run FastAPI with:
# uvicorn backend:app --reload
