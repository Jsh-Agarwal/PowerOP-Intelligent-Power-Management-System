from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from fastapi.responses import JSONResponse
import logging

from ..services.astra_db_service import AstraDBService
from ..auth import oauth2_scheme, validate_token
from ..utils.error_handlers import handle_api_error

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/astra",
    tags=["AstraDB Management"]
)

@router.post("/create_table")
async def create_table(
    table_config: Dict[str, Any],
    token: str = Depends(oauth2_scheme)
):
    """Create a new table in AstraDB."""
    try:
        await validate_token(token)
        
        if not table_config.get("table_name"):
            return JSONResponse(
                status_code=422,
                content={
                    "status": "error",
                    "detail": "table_name is required"
                }
            )
            
        db = AstraDBService()
        await db.connect()
        
        result = await db.create_table(
            table_name=table_config["table_name"],
            schema=table_config.get("schema", {}),
            options=table_config.get("options", {})
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "data": result
            }
        )
    except Exception as e:
        raise handle_api_error(e, "create_table")
