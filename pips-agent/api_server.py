"""
FastAPI server for pips-agent hint generation API.

This module provides HTTP endpoints for the graduated hint system,
allowing the frontend to request contextually-aware hints at 4 levels.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI application
app = FastAPI(
    title="Pips Agent API",
    description="API for graduated hint generation in the pips puzzle system",
    version="1.0.0",
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint for service monitoring."""
    return {"status": "healthy", "service": "pips-agent"}


# Hint generation endpoint will be added in subtask-1-3
