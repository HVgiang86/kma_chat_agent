from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from bson import ObjectId
from app.models.user import PyObjectId

class MessageBase(BaseModel):
    conversation_id: PyObjectId = Field(..., description="Conversation ID this message belongs to")
    user_id: PyObjectId = Field(..., description="User ID who sent this message")
    content: str = Field(..., description="Message content")
    role: Literal["human", "bot"] = Field(..., description="Role of the sender (human or bot)")
    has_attachment: bool = Field(default=False, description="Whether the message has attachments")
    attachments: List[Dict[str, Any]] = Field(default_factory=list, description="List of attachments")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    deleted_at: Optional[datetime] = Field(default=None)

class MessageCreate(BaseModel):
    conversation_id: str
    user_id: str
    content: str
    role: Literal["human", "bot"]
    has_attachment: bool = False
    attachments: List[Dict[str, Any]] = []

class MessageUpdate(BaseModel):
    content: Optional[str] = None
    has_attachment: Optional[bool] = None
    attachments: Optional[List[Dict[str, Any]]] = None

class MessageInDB(MessageBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

    model_config = {
        "json_encoders": {ObjectId: str},
        "populate_by_name": True
    }

class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    user_id: str
    content: str
    role: str
    has_attachment: bool
    attachments: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    
    model_config = {
        "populate_by_name": True
    }

class ChatbotRequest(BaseModel):
    conversation_id: str
    user_id: str
    query: str
    attachments: List[Dict[str, Any]] = []

class AnonymousChatbotRequest(BaseModel):
    query: str
    attachments: List[Dict[str, Any]] = [] 