from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any
import logging

from ..services.groq_slm_service import GroqSLMService
from ..auth import oauth2_scheme, validate_token
from ..utils.error_handlers import handle_api_error
from ..schemas import LLMAnalysisRequest  # Add this import

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/groq",
    tags=["Groq Integration"]
)

@router.get("/context/{context_id}")
async def get_context(
    context_id: str,
    token: str = Depends(oauth2_scheme)
):
    """Get context data for Groq LLM."""
    try:
        await validate_token(token)
        groq = GroqSLMService()
        context = await groq.get_context(context_id)
        return {
            "context_id": context_id,
            "data": context
        }
    except Exception as e:
        raise handle_api_error(e, "get_context")

@router.post("/optimize")
async def optimize_with_groq(
    request: LLMAnalysisRequest,
    token: str = Depends(oauth2_scheme)
):
    """Use Groq LLM for optimization."""
    try:
        await validate_token(token)
        groq = GroqSLMService()
        await groq.connect()
        
        result = await groq.optimize(
            text=request.query,  # Changed from query to text
            context=request.context.model_dump()
        )
        return {
            "status": "success",
            "query": request.query,
            "result": result
        }
    except Exception as e:
        raise handle_api_error(e, "optimize_with_groq")
    finally:
        if groq:
            await groq.close()
