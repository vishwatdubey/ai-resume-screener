from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import candidates, jobs, auth
from app.core.config import settings
from app.db.session import engine
from app.db import models

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Resume Screener",
    description="API for matching candidates with job descriptions",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(auth.router, prefix="/api", tags=["Authentication"])
app.include_router(candidates.router, prefix="/api", tags=["Candidates"])
app.include_router(jobs.router, prefix="/api", tags=["Jobs"])

@app.get("/")
async def root():
    return {"message": "AI Resume Screener API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 