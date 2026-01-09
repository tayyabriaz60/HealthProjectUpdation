"""
Pydantic Models for request (input) and response (output) validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    message: str = Field(..., description="User's message to the chatbot", min_length=1)
    chat_id: Optional[str] = Field(None, description="Chat session ID for multi-turn conversations")
    user_id: Optional[str] = Field(None, description="Firebase user ID (optional, for linking sessions to users)")


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    response: str = Field(..., description="AI-generated response from the chatbot")
    chat_id: str = Field(..., description="Chat session ID for maintaining conversation history")


class ChatSessionCreate(BaseModel):
    """Request schema for creating a new chat session."""
    model: Optional[str] = Field(None, description="Model name (optional, uses default if not provided)")


class ChatSessionResponse(BaseModel):
    """Response schema for chat session creation."""
    chat_id: str = Field(..., description="Unique chat session ID")


class MessageHistory(BaseModel):
    """Schema for a single message in chat history."""
    role: str = Field(..., description="Message role: 'user' or 'model'")
    text: str = Field(..., description="Message content")
    image_path: Optional[str] = Field(
        None,
        description="Optional relative image path if this message is associated with an image",
    )
    audio_path: Optional[str] = Field(
        None,
        description="Optional audio path for voice messages",
    )
    is_voice: Optional[bool] = Field(
        False,
        description="Whether this message is a voice message",
    )


class ChatHistoryResponse(BaseModel):
    """Response schema for chat history."""
    chat_id: str = Field(..., description="Chat session ID")
    history: List[MessageHistory] = Field(..., description="List of messages in the conversation")


class UnifiedChatResponse(BaseModel):
    """Unified response schema that includes both response and history."""
    response: str = Field(..., description="AI-generated response from the chatbot")
    chat_id: str = Field(..., description="Chat session ID for maintaining conversation history")
    history: Optional[List[MessageHistory]] = Field(None, description="Full conversation history (included if requested)")

