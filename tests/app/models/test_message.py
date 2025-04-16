from datetime import datetime
from bson import ObjectId
from app.models.message import (
    MessageBase,
    MessageCreate,
    MessageUpdate,
    MessageInDB,
    MessageResponse,
    ChatbotRequest
)
from app.models.user import PyObjectId

def test_message_base_creation():
    message = MessageBase(
        conversation_id=ObjectId(),
        user_id=ObjectId(),
        content="Test message",
        role="human",
        has_attachment=False,
        attachments=[]
    )
    
    assert isinstance(message.conversation_id, ObjectId)
    assert isinstance(message.user_id, ObjectId)
    assert message.content == "Test message"
    assert message.role == "human"
    assert message.has_attachment is False
    assert message.attachments == []
    assert isinstance(message.created_at, datetime)
    assert isinstance(message.updated_at, datetime)
    assert message.deleted_at is None

def test_message_create():
    message = MessageCreate(
        conversation_id=str(ObjectId()),
        user_id=str(ObjectId()),
        content="Test message",
        role="human"
    )
    
    assert isinstance(message.conversation_id, str)
    assert isinstance(message.user_id, str)
    assert message.content == "Test message"
    assert message.role == "human"
    assert message.has_attachment is False
    assert message.attachments == []

def test_message_update():
    message = MessageUpdate(
        content="Updated message",
        has_attachment=True,
        attachments=[{"type": "file", "url": "test.pdf"}]
    )
    
    assert message.content == "Updated message"
    assert message.has_attachment is True
    assert message.attachments == [{"type": "file", "url": "test.pdf"}]

def test_message_in_db():
    message_id = ObjectId()
    message = MessageInDB(
        _id=message_id,
        conversation_id=ObjectId(),
        user_id=ObjectId(),
        content="Test message",
        role="human"
    )
    
    assert message.id == message_id
    assert isinstance(message.conversation_id, ObjectId)
    assert isinstance(message.user_id, ObjectId)
    assert message.content == "Test message"
    assert message.role == "human"
    assert message.has_attachment is False
    assert message.attachments == []

def test_message_response():
    now = datetime.utcnow()
    message = MessageResponse(
        id=str(ObjectId()),
        conversation_id=str(ObjectId()),
        user_id=str(ObjectId()),
        content="Test message",
        role="human",
        has_attachment=False,
        attachments=[],
        created_at=now,
        updated_at=now
    )
    
    assert isinstance(message.id, str)
    assert isinstance(message.conversation_id, str)
    assert isinstance(message.user_id, str)
    assert message.content == "Test message"
    assert message.role == "human"
    assert message.has_attachment is False
    assert message.attachments == []
    assert message.created_at == now
    assert message.updated_at == now

def test_chatbot_request():
    request = ChatbotRequest(
        conversation_id=str(ObjectId()),
        user_id=str(ObjectId()),
        query="Test query",
        attachments=[{"type": "file", "url": "test.pdf"}]
    )
    
    assert isinstance(request.conversation_id, str)
    assert isinstance(request.user_id, str)
    assert request.query == "Test query"
    assert request.attachments == [{"type": "file", "url": "test.pdf"}] 