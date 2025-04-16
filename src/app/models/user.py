from datetime import datetime
from typing import Optional, Dict, Any, ClassVar, Annotated
from pydantic import BaseModel, Field, GetJsonSchemaHandler
from bson import ObjectId
from typing_extensions import Self
import json
from pydantic.json_schema import JsonSchemaValue

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        # For backward compatibility - this is deprecated in Pydantic v2
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        from pydantic_core import core_schema
        
        return core_schema.union_schema([
            # First try to validate as ObjectId directly
            core_schema.is_instance_schema(ObjectId),
            # If not an ObjectId, try to convert from string
            core_schema.chain_schema([
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(cls.validate),
            ]),
        ])
        
    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {"type": "string"}

class UserBase(BaseModel):
    student_code: str = Field(..., description="Student's unique code")
    name: Optional[str] = Field(None, description="Student's full name")
    student_class: Optional[str] = Field(None, description="Student's class")
    app_settings: Dict = Field(default_factory=dict, description="User's app settings")
    is_guest: bool = Field(default=False, description="Whether the user is a guest")
    is_active: bool = Field(default=True, description="Whether the user is active")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    deleted_at: Optional[datetime] = Field(default=None)

class UserCreate(BaseModel):
    student_code: str
    name: Optional[str] = None
    student_class: Optional[str] = None
    app_settings: Optional[Dict] = None
    is_guest: Optional[bool] = None

class UserUpdate(BaseModel):
    name: Optional[str] = None
    student_class: Optional[str] = None
    app_settings: Optional[Dict] = None
    is_active: Optional[bool] = None

class UserInDB(UserBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

    model_config = {
        "json_encoders": {ObjectId: str},
        "populate_by_name": True
    }

class UserResponse(BaseModel):
    id: str
    student_code: str
    name: Optional[str] = None
    student_class: Optional[str] = None
    app_settings: Optional[Dict] = None
    is_guest: bool
    is_active: bool
    
    model_config = {
        "populate_by_name": True
    } 