"""Pydantic models for request/response validation."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str
    model_loaded: bool
    model_info: dict


class RootResponse(BaseModel):
    """Response model for root endpoint."""
    
    name: str
    version: str
    endpoints: dict[str, str]
