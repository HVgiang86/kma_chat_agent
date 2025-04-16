from datetime import datetime
from bson import ObjectId
from app.models.conversation import (
    ConversationBase,
    ConversationCreate,
    ConversationUpdate,
    ConversationInDB,
    ConversationResponse
)

def test_conversation_base_creation():
    conversation = ConversationBase(
        name="Test Conversation",
        user_id=ObjectId(),
        is_active=True,
        is_shared=False,
        share_token=None
    )
    
    assert conversation.name == "Test Conversation"
    assert isinstance(conversation.user_id, ObjectId)
    assert conversation.is_active is True
    assert conversation.is_shared is False
    assert conversation.share_token is None
    assert isinstance(conversation.created_at, datetime)
    assert isinstance(conversation.updated_at, datetime)
    assert conversation.deleted_at is None

def test_conversation_create():
    conversation = ConversationCreate(
        name="Test Conversation",
        user_id=str(ObjectId())
    )
    
    assert conversation.name == "Test Conversation"
    assert isinstance(conversation.user_id, str)

def test_conversation_update():
    conversation = ConversationUpdate(
        name="Updated Conversation",
        is_active=False,
        is_shared=True,
        share_token="test_token"
    )
    
    assert conversation.name == "Updated Conversation"
    assert conversation.is_active is False
    assert conversation.is_shared is True
    assert conversation.share_token == "test_token"

def test_conversation_in_db():
    conversation_id = ObjectId()
    conversation = ConversationInDB(
        _id=conversation_id,
        name="Test Conversation",
        user_id=ObjectId(),
        is_active=True,
        is_shared=False
    )
    
    assert conversation.id == conversation_id
    assert conversation.name == "Test Conversation"
    assert isinstance(conversation.user_id, ObjectId)
    assert conversation.is_active is True
    assert conversation.is_shared is False
    assert conversation.share_token is None

def test_conversation_response():
    now = datetime.utcnow()
    conversation = ConversationResponse(
        id=str(ObjectId()),
        name="Test Conversation",
        user_id=str(ObjectId()),
        is_active=True,
        is_shared=False,
        share_token=None,
        created_at=now,
        updated_at=now
    )
    
    assert isinstance(conversation.id, str)
    assert conversation.name == "Test Conversation"
    assert isinstance(conversation.user_id, str)
    assert conversation.is_active is True
    assert conversation.is_shared is False
    assert conversation.share_token is None
    assert conversation.created_at == now
    assert conversation.updated_at == now 