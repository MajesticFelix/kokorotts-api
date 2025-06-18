from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from .tts_engine import synthesize, synthesize_with_voice_blend, synthesize_streaming, synthesize_streaming_with_voice_blend
from typing import Dict, Optional, Union, List
from huggingface_hub import list_repo_files
import io
import re

app = FastAPI(
    title="Kokoro TTS API", 
    version="1.0.0",
    description="OpenAI-compatible TTS API using Kokoro model with voice blending support"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Kokoro TTS API is running!", "docs": "/docs", "openai_endpoint": "/v1/audio/speech", "test_page": "/test"}

@app.get("/test", response_class=HTMLResponse)
async def test_page():
    with open("static/test_streaming.html", "r") as f: # currently not implemented
        return HTMLResponse(content=f.read())


# OpenAI Compatible API Models
class OpenAISpeechRequest(BaseModel):
    model: str = Field(default="kokoro", description="TTS model to use")
    input: str = Field(..., description="Text to convert to speech")
    voice: str = Field(default="af_heart", description="Voice name or blended voices (e.g., 'af_bella+af_sky' or 'af_bella(2)+af_sky(1)')")
    response_format: Optional[str] = Field(default="mp3", description="Audio format: wav, mp3, flac, ogg, opus")
    speed: Optional[float] = Field(default=1.0, ge=0.25, le=4.0, description="Speech speed (0.25 to 4.0)")
    stream: Optional[bool] = Field(default=False, description="Enable streaming response")

class OpenAIVoiceResponse(BaseModel):
    voices: List[str]


# Create OpenAI compatible router
openai_router = APIRouter(prefix="/v1")

def parse_voice_specification(voice_spec: str) -> Union[str, Dict[str, float]]:
    """Parse OpenAI voice specification into voice name or voice weights dict
    
    Examples:
    - 'af_bella' -> 'af_bella'
    - 'af_bella+af_sky' -> {'af_bella': 0.5, 'af_sky': 0.5}
    - 'af_bella(2)+af_sky(1)' -> {'af_bella': 0.67, 'af_sky': 0.33}
    """
    if '+' not in voice_spec:
        return voice_spec.strip()
    
    # Parse weighted voice combinations
    voice_weights = {}
    voices = voice_spec.split('+')
    
    for voice in voices:
        voice = voice.strip()
        # Check for weight specification: voice_name(weight)
        weight_match = re.match(r'(.+?)\(([0-9.]+)\)$', voice)
        if weight_match:
            voice_name = weight_match.group(1).strip()
            weight = float(weight_match.group(2))
        else:
            voice_name = voice
            weight = 1.0
        
        voice_weights[voice_name] = weight
    
    return voice_weights

def get_media_type(format_name: str) -> str:
    """Get proper media type for audio format"""
    media_types = {
        "wav": "audio/wav",
        "flac": "audio/flac", 
        "ogg": "audio/ogg",
        "mp3": "audio/mpeg",
        "opus": "audio/opus"
    }
    return media_types.get(format_name.lower(), "audio/mpeg")

@openai_router.post("/audio/speech")
async def create_speech(request: OpenAISpeechRequest):
    """OpenAI compatible speech synthesis endpoint"""
    # Validate format first (before try block to avoid wrapping HTTPException)
    if request.response_format not in ["wav", "mp3", "flac", "ogg", "opus"]:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": f"Unsupported response_format: {request.response_format}", "type": "invalid_request_error"}}
        )
    
    # Validate input text
    if not request.input or not request.input.strip():
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "Input text cannot be empty", "type": "invalid_request_error"}}
        )
    
    # Validate speed parameter
    if not (0.25 <= request.speed <= 4.0):
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "Speed must be between 0.25 and 4.0", "type": "invalid_request_error"}}
        )
    
    try:
        
        # Parse voice specification
        voice_spec = parse_voice_specification(request.voice)
        
        # Set media type and filename
        media_type = get_media_type(request.response_format)
        
        if request.stream:
            # Streaming response
            def generate_audio():
                if isinstance(voice_spec, dict):
                    # Voice blending with streaming
                    for chunk in synthesize_streaming_with_voice_blend(
                        request.input, voice_spec, request.speed, request.response_format
                    ):
                        yield chunk
                else:
                    # Single voice streaming
                    for chunk in synthesize_streaming(
                        request.input, voice_spec, request.speed, request.response_format
                    ):
                        yield chunk
            
            headers = {
                "Content-Type": media_type,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
            return StreamingResponse(generate_audio(), media_type=media_type, headers=headers)
        else:
            # Non-streaming response
            if isinstance(voice_spec, dict):
                # Voice blending
                audio_bytes = synthesize_with_voice_blend(
                    request.input, voice_spec, request.speed, request.response_format
                )
            else:
                # Single voice
                audio_bytes = synthesize(
                    request.input, voice_spec, request.speed, request.response_format
                )
            
            audio_stream = io.BytesIO(audio_bytes)
            filename = f"speech.{request.response_format}"
            headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
            return StreamingResponse(audio_stream, media_type=media_type, headers=headers)
            
    except ValueError as e:
        # Handle voice parsing and synthesis validation errors
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": str(e), "type": "invalid_request_error"}}
        )
    except RuntimeError as e:
        # Handle TTS synthesis errors
        error_msg = str(e)
        if "synthesis failed" in error_msg.lower() or "voice" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": error_msg, "type": "invalid_request_error"}}
            )
        else:
            raise HTTPException(
                status_code=500,
                detail={"error": {"message": f"Synthesis error: {error_msg}", "type": "internal_server_error"}}
            )
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": f"Internal server error: {str(e)}", "type": "internal_server_error"}}
        )

@openai_router.get("/audio/voices", response_model=OpenAIVoiceResponse)
async def list_voices():
    """OpenAI compatible voice listing endpoint"""
    try:
        files = list_repo_files(repo_id="hexgrad/Kokoro-82M")
        voice_files = [f for f in files if f.endswith(".pt") and "voices" in f]
        voices = [f.split("/")[-1].replace(".pt", "") for f in voice_files]
        return OpenAIVoiceResponse(voices=voices)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": f"Failed to list voices: {str(e)}", "type": "internal_server_error"}}
        )

# Include OpenAI router in main app
app.include_router(openai_router)