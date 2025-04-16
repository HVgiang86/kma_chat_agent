import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from bson import ObjectId
from app.services.chatbot import (
    get_conversation,
    send_query_to_agent,
    process_chat_request
)

@pytest.fixture
def mock_conversation():
    return MagicMock(return_value={"answer": "Test response"})

@pytest.fixture
def mock_db():
    with patch("app.services.chatbot.Database") as mock:
        mock.insert_one = AsyncMock(return_value=ObjectId())
        mock.update_one = AsyncMock()
        mock.get_object_id = MagicMock(return_value=ObjectId())
        yield mock

@pytest.mark.asyncio
async def test_get_conversation(mock_conversation):
    with patch("app.services.chatbot.initialize_rag", return_value=mock_conversation):
        conversation = await get_conversation()
        assert conversation == mock_conversation

@pytest.mark.asyncio
async def test_send_query_to_agent_success(mock_conversation):
    with patch("app.services.chatbot.get_conversation", return_value=mock_conversation):
        response = await send_query_to_agent("test query")
        assert response == "Test response"
        mock_conversation.assert_called_once_with({"question": "test query"})

@pytest.mark.asyncio
async def test_send_query_to_agent_error():
    with patch("app.services.chatbot.get_conversation", side_effect=Exception("Test error")):
        response = await send_query_to_agent("test query")
        assert response == "I apologize, but I encountered an error while processing your query."

@pytest.mark.asyncio
async def test_process_chat_request_success(mock_conversation, mock_db):
    with patch("app.services.chatbot.get_conversation", return_value=mock_conversation):
        conversation_id = str(ObjectId())
        user_id = str(ObjectId())
        query = "test query"
        
        result = await process_chat_request(conversation_id, user_id, query)
        
        # Verify database calls
        assert mock_db.insert_one.call_count == 2  # Called for both user and bot messages
        mock_db.update_one.assert_called_once()
        
        # Verify response structure
        assert "user_message" in result
        assert "bot_message" in result
        assert result["user_message"]["content"] == query
        assert result["bot_message"]["content"] == "Test response"

@pytest.mark.asyncio
async def test_process_chat_request_with_attachments(mock_conversation, mock_db):
    with patch("app.services.chatbot.get_conversation", return_value=mock_conversation):
        conversation_id = str(ObjectId())
        user_id = str(ObjectId())
        query = "test query"
        attachments = [{"type": "file", "url": "test.pdf"}]
        
        result = await process_chat_request(conversation_id, user_id, query, attachments)
        
        # Verify database calls
        assert mock_db.insert_one.call_count == 2
        mock_db.update_one.assert_called_once()
        
        # Verify attachments in user message
        assert result["user_message"]["has_attachment"] is True
        assert result["user_message"]["attachments"] == attachments
        assert result["bot_message"]["has_attachment"] is False

@pytest.mark.asyncio
async def test_process_chat_request_error(mock_conversation, mock_db):
    with patch("app.services.chatbot.get_conversation", side_effect=Exception("Test error")):
        conversation_id = str(ObjectId())
        user_id = str(ObjectId())
        query = "test query"
        
        with pytest.raises(Exception):
            await process_chat_request(conversation_id, user_id, query) 