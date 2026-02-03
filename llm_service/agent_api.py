#!/usr/bin/env python3
"""
Surgical Command Agent API

A FastAPI service that accepts text commands (from ASR or XR headsets)
and returns structured tool calls for medical device control.

Architecture:
- Receives text input via REST API
- Sends to local vLLM with tool definitions
- Returns structured tool calls or clarification requests
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
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct-AWQ")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))

app = FastAPI(
    title="Surgical Command Agent",
    description="Voice-to-device-command agent for surgical environments",
    version="0.1.0"
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
# Tool Definitions - Define available medical device controls
# =============================================================================

SURGICAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "adjust_surgical_light",
            "description": "Adjust the surgical overhead light. Can change intensity (brightness) and position.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intensity": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Light intensity as percentage (0-100)"
                    },
                    "position": {
                        "type": "string",
                        "enum": ["center", "left", "right", "patient_head", "patient_torso", "patient_legs"],
                        "description": "Where to aim the light"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_operating_table",
            "description": "Adjust the operating table height, tilt, or rotation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "height_change_cm": {
                        "type": "integer",
                        "minimum": -30,
                        "maximum": 30,
                        "description": "Height adjustment in centimeters (positive=up, negative=down)"
                    },
                    "tilt_degrees": {
                        "type": "integer",
                        "minimum": -45,
                        "maximum": 45,
                        "description": "Tilt angle in degrees (positive=head up, negative=head down)"
                    },
                    "lateral_tilt_degrees": {
                        "type": "integer",
                        "minimum": -20,
                        "maximum": 20,
                        "description": "Lateral tilt in degrees (positive=right side up)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "control_room_environment",
            "description": "Control room environment including ambient lighting, temperature, and music.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ambient_light_percent": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Ambient room lighting percentage"
                    },
                    "temperature_celsius": {
                        "type": "number",
                        "minimum": 16,
                        "maximum": 26,
                        "description": "Room temperature in Celsius"
                    },
                    "music_action": {
                        "type": "string",
                        "enum": ["play", "pause", "stop", "next", "previous", "volume_up", "volume_down"],
                        "description": "Music playback control"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "control_display",
            "description": "Control displays and monitors in the surgical suite.",
            "parameters": {
                "type": "object",
                "properties": {
                    "display_id": {
                        "type": "string",
                        "enum": ["main", "secondary", "anesthesia", "nursing"],
                        "description": "Which display to control"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["show_vitals", "show_imaging", "show_camera", "show_notes", "blank", "pip_toggle"],
                        "description": "What to display or action to take"
                    },
                    "source": {
                        "type": "string",
                        "description": "Optional: specific source (e.g., 'ct_scan_1', 'endoscope_2')"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "request_assistance",
            "description": "Request assistance or alert team members.",
            "parameters": {
                "type": "object",
                "properties": {
                    "urgency": {
                        "type": "string",
                        "enum": ["routine", "urgent", "emergency"],
                        "description": "Urgency level of the request"
                    },
                    "role": {
                        "type": "string",
                        "enum": ["nurse", "anesthesiologist", "surgeon", "technician", "all"],
                        "description": "Who to alert"
                    },
                    "message": {
                        "type": "string",
                        "description": "Optional message to include with the alert"
                    }
                },
                "required": ["urgency", "role"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "control_insufflator",
            "description": "Control the CO2 insufflator for laparoscopic procedures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pressure_mmhg": {
                        "type": "integer",
                        "minimum": 8,
                        "maximum": 20,
                        "description": "Target intra-abdominal pressure in mmHg"
                    },
                    "flow_rate": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "max"],
                        "description": "CO2 flow rate"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["start", "stop", "desufflate"],
                        "description": "Insufflator action"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_recording",
            "description": "Start or stop surgical recording.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["start", "stop", "pause", "resume", "snapshot"],
                        "description": "Recording action"
                    },
                    "source": {
                        "type": "string",
                        "enum": ["endoscope", "room_camera", "all"],
                        "description": "Which video source to record"
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional label/bookmark for this recording segment"
                    }
                },
                "required": ["action"]
            }
        }
    }
]

# System prompt for the surgical assistant
SYSTEM_PROMPT = """You are a surgical room command interpreter. Your role is to:

1. Listen to voice commands from the surgical team
2. Interpret commands and map them to available device controls
3. Return precise, structured tool calls

IMPORTANT GUIDELINES:
- Be precise with parameters - surgical environments require accuracy
- If a command is ambiguous, make reasonable assumptions based on context
- For safety-critical actions (like emergency alerts), confirm the intent
- You can call multiple tools in parallel when appropriate
- If a command doesn't map to any available tool, explain what you can do instead

CONTEXT:
- You're in an active surgical suite
- Commands may be informal ("more light", "table up a bit")
- Interpret natural language into specific parameters
- "A bit" typically means small adjustments (10-20%)
- "A lot" or "much more" means larger adjustments (30-50%)

Always respond helpfully even if you can't execute the exact request."""


# =============================================================================
# Request/Response Models
# =============================================================================

class CommandRequest(BaseModel):
    """Incoming command from ASR or XR headset"""
    text: str = Field(..., description="The transcribed command text")
    context: Optional[dict] = Field(default=None, description="Optional context (e.g., current procedure, user role)")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation continuity")


class ToolCall(BaseModel):
    """A single tool call to execute"""
    id: str
    name: str
    arguments: dict


class CommandResponse(BaseModel):
    """Response containing tool calls and/or message"""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    message: Optional[str] = Field(default=None, description="Text response if no tools called or for confirmation")
    needs_confirmation: bool = Field(default=False, description="Whether this action needs user confirmation")
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
    # Keep only last 10 exchanges
    if len(history) > 20:
        conversation_history[session_id] = history[-20:]


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
        tools_available=len(SURGICAL_TOOLS)
    )


@app.get("/tools")
async def list_tools():
    """List all available tools/device controls"""
    return {
        "tools": SURGICAL_TOOLS,
        "count": len(SURGICAL_TOOLS)
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
            tools=SURGICAL_TOOLS,
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

        # Check if confirmation needed (safety-critical actions)
        needs_confirmation = any(
            tc.name == "request_assistance" and
            tc.arguments.get("urgency") == "emergency"
            for tc in tool_calls
        )

        # Update history
        add_to_history(session_id, "user", request.text)
        if message:
            add_to_history(session_id, "assistant", message)

        processing_time = (time.time() - start_time) * 1000

        return CommandResponse(
            tool_calls=tool_calls,
            message=message,
            needs_confirmation=needs_confirmation,
            session_id=session_id,
            processing_time_ms=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


@app.post("/command/execute")
async def execute_command(request: CommandRequest):
    """
    Process command and simulate tool execution.

    In production, this would actually call device APIs.
    For now, returns what would be executed.
    """
    # First, get the tool calls
    response = await process_command(request)

    # Simulate execution
    execution_results = []
    for tc in response.tool_calls:
        execution_results.append({
            "tool": tc.name,
            "arguments": tc.arguments,
            "status": "simulated",
            "result": f"Would execute {tc.name} with {tc.arguments}"
        })

    return {
        "command": request.text,
        "tool_calls": response.tool_calls,
        "execution_results": execution_results,
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

    parser = argparse.ArgumentParser(description="Surgical Command Agent API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print("Surgical Command Agent API")
    print(f"{'='*50}")
    print(f"vLLM Backend: {VLLM_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Tools available: {len(SURGICAL_TOOLS)}")
    print(f"{'='*50}\n")

    uvicorn.run(app, host=args.host, port=args.port)
