from fastapi import Depends, HTTPException, status
from .auth import oauth2_scheme, validate_token
from .models import User

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Dependency to get current authenticated user."""
    try:
        payload = await validate_token(token)
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
        return User(username=username)
    except HTTPException as e:
        raise e
