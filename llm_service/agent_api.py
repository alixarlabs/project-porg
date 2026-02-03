#!/usr/bin/env python3
"""
Camera Control Agent API

A FastAPI service that accepts voice commands and returns structured
tool calls for surgical camera PTZ control (project-jango integration).

Architecture:
- Receives text input via REST API
- Sends to local vLLM with tool definitions
- Returns structured tool calls matching project-jango JSON protocol
"""

import os
import json
import asyncio
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

# Configuration
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8889/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))

app = FastAPI(
    title="Camera Control Agent",
    description="Voice-to-camera-command agent for surgical endoscope control",
    version="0.2.0"
)

# CORS for XR headsets and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# vLLM client (OpenAI-compatible)
client = AsyncOpenAI(
    base_url=VLLM_BASE_URL,
    api_key="not-needed"  # vLLM doesn't require API key
)


# =============================================================================
# Tool Definitions - Camera PTZ Controls (project-jango compatible)
# =============================================================================

CAMERA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "camera_move",
            "description": "Move the surgical camera in a direction. Supports 8-way directional movement for pan and tilt control.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right", "up_left", "up_right", "down_left", "down_right"],
                        "description": "Direction to move the camera"
                    }
                },
                "required": ["direction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "camera_zoom",
            "description": "Zoom the surgical camera in or out. Use 'in' to magnify/get closer, 'out' to zoom out/see more.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["in", "out"],
                        "description": "Zoom direction: 'in' to magnify, 'out' to widen view"
                    }
                },
                "required": ["direction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "camera_focus",
            "description": "Adjust the camera focus. Use 'in' for closer focus (near objects), 'out' for farther focus (distant objects).",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["in", "out"],
                        "description": "Focus direction: 'in' for closer, 'out' for farther"
                    }
                },
                "required": ["direction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "camera_stop",
            "description": "Stop all camera movement immediately. Use for emergency stop or when the desired position is reached.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_action",
            "description": "Trigger a surgical instrument action. Actions 1-6 control different instrument functions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3, 4, 5, 6],
                        "description": "Action number: 0=none/off, 1-6=specific instrument actions"
                    }
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "change_mode",
            "description": "Change the camera control mode. Switches which input source controls the camera.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["headset", "gamepad", "tool", "gui", "test_pattern"],
                        "description": "Control mode: 'headset' for voice/XR, 'gamepad' for controller, 'tool' for AI tracking, 'gui' for web interface, 'test_pattern' for automated testing"
                    }
                },
                "required": ["mode"]
            }
        }
    }
]

# System prompt for the camera control assistant
SYSTEM_PROMPT = """You are a surgical camera control assistant. Your role is to interpret voice commands and control a surgical endoscope camera.

AVAILABLE CONTROLS:
- camera_move: Pan/tilt the camera (up, down, left, right, or diagonals)
- camera_zoom: Zoom in (magnify) or out (wider view)
- camera_focus: Adjust focus closer or farther
- camera_stop: Stop all movement immediately
- trigger_action: Trigger instrument actions (1-6)
- change_mode: Switch control source (headset, gamepad, tool, gui)

COMMAND INTERPRETATION:
- "move up/down/left/right" → camera_move with that direction
- "zoom in/out" or "closer/farther view" → camera_zoom
- "focus" or "sharpen" or "blur" → camera_focus
- "stop" or "hold" or "freeze" → camera_stop
- "action 1/2/3..." or instrument names → trigger_action
- "switch to gamepad/manual/voice" → change_mode

IMPORTANT:
- Be concise - just call the appropriate tool
- For diagonal movement, use compound directions (up_left, down_right, etc.)
- "pan left" = camera_move left, "tilt up" = camera_move up
- If unclear, prefer camera_move for spatial commands
- Only respond with text if the command cannot be mapped to a tool"""


# =============================================================================
# Request/Response Models
# =============================================================================

class CommandRequest(BaseModel):
    """Incoming command from ASR or XR headset"""
    text: str = Field(..., description="The transcribed command text")
    context: Optional[dict] = Field(default=None, description="Optional context")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation continuity")


class ToolCall(BaseModel):
    """A single tool call to execute"""
    id: str
    name: str
    arguments: dict


class CommandResponse(BaseModel):
    """Response containing tool calls and/or message"""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    message: Optional[str] = Field(default=None, description="Text response if no tools called")
    needs_confirmation: bool = Field(default=False)
    session_id: Optional[str] = None
    processing_time_ms: float = 0


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model: str
    vllm_url: str
    tools_available: int


# =============================================================================
# Conversation History (simple in-memory store)
# =============================================================================

conversation_history: dict[str, list] = {}


def get_history(session_id: str) -> list:
    """Get conversation history for a session"""
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    return conversation_history[session_id]


def add_to_history(session_id: str, role: str, content: str):
    """Add a message to conversation history"""
    history = get_history(session_id)
    history.append({"role": role, "content": content})
    # Keep only last 6 exchanges (shorter for faster context)
    if len(history) > 12:
        conversation_history[session_id] = history[-12:]


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model=MODEL_NAME,
        vllm_url=VLLM_BASE_URL,
        tools_available=len(CAMERA_TOOLS)
    )


@app.get("/tools")
async def list_tools():
    """List all available tools/device controls"""
    return {
        "tools": CAMERA_TOOLS,
        "count": len(CAMERA_TOOLS)
    }


@app.post("/command", response_model=CommandResponse)
async def process_command(request: CommandRequest):
    """
    Process a voice command and return tool calls.

    This is the main endpoint for ASR integration.
    """
    import time
    start_time = time.time()

    session_id = request.session_id or "default"

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history for context
    messages.extend(get_history(session_id))

    # Add current command
    user_message = request.text
    if request.context:
        user_message = f"[Context: {json.dumps(request.context)}]\n\n{request.text}"

    messages.append({"role": "user", "content": user_message})

    try:
        # Call vLLM with tools
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=CAMERA_TOOLS,
            tool_choice="auto",
            max_tokens=MAX_TOKENS,
            temperature=0.1  # Low temperature for consistent tool calls
        )

        choice = response.choices[0]

        # Parse tool calls
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                ))

        # Get any text message
        message = choice.message.content

        # Update history
        add_to_history(session_id, "user", request.text)
        if message:
            add_to_history(session_id, "assistant", message)

        processing_time = (time.time() - start_time) * 1000

        return CommandResponse(
            tool_calls=tool_calls,
            message=message,
            needs_confirmation=False,
            session_id=session_id,
            processing_time_ms=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


@app.post("/command/execute")
async def execute_command(request: CommandRequest):
    """
    Process command and return tool execution format.

    Returns commands in project-jango JSON format ready for UDP transmission.
    """
    # First, get the tool calls
    response = await process_command(request)

    # Convert to project-jango format
    jango_commands = []
    for tc in response.tool_calls:
        if tc.name == "camera_move":
            jango_commands.append({
                "type": "direction",
                "direction": tc.arguments.get("direction", "none")
            })
        elif tc.name == "camera_zoom":
            jango_commands.append({
                "type": "zoom_focus",
                "zoom": tc.arguments.get("direction")
            })
        elif tc.name == "camera_focus":
            jango_commands.append({
                "type": "zoom_focus",
                "focus": tc.arguments.get("direction")
            })
        elif tc.name == "camera_stop":
            jango_commands.append({
                "type": "stop"
            })
        elif tc.name == "trigger_action":
            jango_commands.append({
                "type": "action",
                "action": tc.arguments.get("action", 0)
            })
        elif tc.name == "change_mode":
            jango_commands.append({
                "type": "mode_change",
                "mode": tc.arguments.get("mode", "headset")
            })

    return {
        "command": request.text,
        "tool_calls": response.tool_calls,
        "jango_commands": jango_commands,
        "message": response.message,
        "processing_time_ms": response.processing_time_ms
    }


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    if session_id in conversation_history:
        del conversation_history[session_id]
    return {"status": "cleared", "session_id": session_id}


# =============================================================================
# WebSocket for streaming (optional, for real-time updates)
# =============================================================================

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time command processing.

    Useful for XR headsets that want bidirectional communication.
    """
    await websocket.accept()

    try:
        while True:
            # Receive command
            data = await websocket.receive_json()

            # Process
            request = CommandRequest(
                text=data.get("text", ""),
                context=data.get("context"),
                session_id=session_id
            )

            response = await process_command(request)

            # Send response
            await websocket.send_json({
                "tool_calls": [tc.model_dump() for tc in response.tool_calls],
                "message": response.message,
                "needs_confirmation": response.needs_confirmation,
                "processing_time_ms": response.processing_time_ms
            })

    except WebSocketDisconnect:
        pass


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Camera Control Agent API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8887, help="Port to bind to")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print("Camera Control Agent API")
    print(f"{'='*50}")
    print(f"vLLM Backend: {VLLM_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Tools available: {len(CAMERA_TOOLS)}")
    print(f"{'='*50}\n")

    uvicorn.run(app, host=args.host, port=args.port)
