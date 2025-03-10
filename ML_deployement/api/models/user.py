from pydantic import BaseModel

class User(BaseModel):
    """User model for authentication and user management."""
    username: str
    disabled: bool = False

    class Config:
        from_attributes = True
