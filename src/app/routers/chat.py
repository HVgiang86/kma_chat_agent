from fastapi import APIRouter, HTTPException, status, Depends, Body
from typing import Dict, Any
from bson import ObjectId
from app.models.message import ChatbotRequest, AnonymousChatbotRequest
from app.models.conversation import ConversationCreate
from app.services.chatbot import process_chat_request, send_query_to_agent
from app.db import Database
import uuid

router = APIRouter()

@router.post("/query", status_code=status.HTTP_200_OK)
async def chat_with_bot(request: ChatbotRequest):
    """
    Send a query to the chatbot and get a response
    This will create a message from the user and a response from the bot
    """
    try:
        # Validate conversation and user exist
        conversation = await Database.find_one(
            "conversations", {"_id": ObjectId(request.conversation_id)}
        )
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Conversation not found"
            )
            
        user = await Database.find_one("users", {"_id": ObjectId(request.user_id)})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Check if user owns the conversation
        if str(conversation["user_id"]) != request.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this conversation"
            )
        
        # Process the chat request
        result = await process_chat_request(
            conversation_id=request.conversation_id,
            user_id=request.user_id,
            query=request.query,
            attachments=request.attachments
        )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )

@router.post("/anonymous", status_code=status.HTTP_200_OK)
async def anonymous_chat(request: AnonymousChatbotRequest):
    """
    Send a query to the chatbot without requiring an account
    This will automatically create a conversation and handle the chat
    """
    try:

        response = await send_query_to_agent(
            request.query,
        )
        return {
            "response": response,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing anonymous request: {str(e)}"
        ) 