from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from .tts_engine import _pipelines, _device, synthesize, synthesize_with_voice_blend, synthesize_streaming, synthesize_streaming_with_voice_blend
from typing import Dict, Optional, Union, List
from huggingface_hub import list_repo_files
import io
import re
import psutil
import torch
import time
from datetime import datetime

# Global startup time tracker
startup_time = time.time()

tags_metadata = [
    {
        "name": "Root",
        "description": "Main API information and navigation"
    },
    {
        "name": "Audio",
        "description": "Text-to-speech synthesis and voice management"
    },
    {
        "name": "Testing",
        "description": "Test interface and development tools"
    },
    {
        "name": "Monitoring", 
        "description": "Health checks and system metrics"
    },
    {
        "name": "Debug",
        "description": "Debugging and diagnostic information"
    }
]

app = FastAPI(
    title="Kokoro TTS API", 
    version="1.0.0",
    description="OpenAI-compatible TTS API using Kokoro model with voice blending support",
    openapi_tags=tags_metadata
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

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Kokoro TTS API is running!", 
        "docs": "/docs", 
        "openai_endpoint": "/v1/audio/speech", 
        "voices_endpoint": "/v1/audio/voices", 
        "languages_endpoint": "/v1/audio/languages",
        "test_page": "/test",
        "monitoring": {
            "health": "/health",
            "metrics": "/metrics", 
            "pipeline_status": "/pipeline/status",
            "debug": "/debug"
        }
    }

@app.get("/test", response_class=HTMLResponse, tags=["Testing"])
async def test_page():
    with open("static/test.html", "r") as f:
        return HTMLResponse(content=f.read())

# OpenAI Compatible API Models
class OpenAISpeechRequest(BaseModel):
    model: str = Field(default="kokoro", description="TTS model to use")
    input: str = Field(..., description="Text to convert to speech")
    voice: str = Field(default="af_heart", description="Voice name or blended voices (e.g., 'af_bella+af_sky' or 'af_bella(2)+af_sky(1)')")
    response_format: Optional[str] = Field(default="mp3", description="Audio format: wav, mp3, flac, ogg, opus")
    speed: Optional[float] = Field(default=1.0, ge=0.25, le=4.0, description="Speech speed (0.25 to 4.0)")
    stream: Optional[bool] = Field(default=False, description="Enable streaming response")
    include_captions: Optional[bool] = Field(default=False, description="Include per-word timing information in response")
    language: Optional[str] = Field(default="a", description="Language code (a=American English, b=British English, j=Japanese, z=Chinese, e=Spanish, f=French, h=Hindi, i=Italian, p=Portuguese)")

class OpenAIVoiceResponse(BaseModel):
    voices: List[str]

class OpenAILanguageResponse(BaseModel):
    languages: List[Dict[str, str]]

# Debug and Monitoring Response Models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float
    version: str

class SystemMetricsResponse(BaseModel):
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    gpu_available: bool
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None

class PipelineStatusResponse(BaseModel):
    loaded_languages: List[str]
    default_language: str
    device: str
    pipeline_count: int
    model_info: Dict[str, str]

class DebugInfoResponse(BaseModel):
    system_info: Dict[str, Union[str, float, bool]]
    pipeline_status: Dict[str, Union[str, int, List[str]]]
    recent_performance: Dict[str, Union[str, float]]

# Timestamped Caption Models
class WordTimestamp(BaseModel):
    text: str = Field(..., description="The word text")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    phonemes: Optional[str] = Field(None, description="Phonetic representation")

class CaptionData(BaseModel):
    words: List[WordTimestamp] = Field(..., description="List of words with timestamps")
    total_duration: float = Field(..., description="Total audio duration in seconds")
    sample_rate: int = Field(default=24000, description="Audio sample rate")

class CaptionedSpeechResponse(BaseModel):
    audio_url: Optional[str] = Field(None, description="URL to download audio file")
    captions: CaptionData = Field(..., description="Word-level timing information")
    metadata: Dict[str, Union[str, float, int]] = Field(..., description="Generation metadata")

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

def get_supported_voices() -> List[str]:
    """Get list of supported voices from repository"""
    files = list_repo_files(repo_id="hexgrad/Kokoro-82M")
    voice_files = [f for f in files if f.endswith(".pt") and "voices" in f]
    voices = [f.split("/")[-1].replace(".pt", "") for f in voice_files]
    return voices

def get_language_map() -> Dict[str, Dict[str, str]]:
    """Get the mapping of language codes to language information"""
    return {
        "a": {"name": "American English", "iso": "en-US"},
        "b": {"name": "British English", "iso": "en-GB"},
        "j": {"name": "Japanese", "iso": "ja-JP"},
        "z": {"name": "Mandarin Chinese", "iso": "zh-CN"},
        "e": {"name": "Spanish", "iso": "es-ES"},
        "f": {"name": "French", "iso": "fr-FR"},
        "h": {"name": "Hindi", "iso": "hi-IN"},
        "i": {"name": "Italian", "iso": "it-IT"},
        "p": {"name": "Brazilian Portuguese", "iso": "pt-BR"}
    }

def get_supported_languages() -> List[Dict[str, str]]:
    """Infer supported languages from actual voice files in repository"""
    try:
        # Get voice files from repository
        files = list_repo_files(repo_id="hexgrad/Kokoro-82M")
        voice_files = [f for f in files if f.endswith(".pt") and "voices" in f]
        
        # Language mapping based on official Kokoro documentation
        language_map = get_language_map()
        
        # Extract language codes from voice file names
        found_lang_codes = set()
        for voice_file in voice_files:
            voice_name = voice_file.split("/")[-1].replace(".pt", "")
            if "_" in voice_name:
                # Extract first character as language code (e.g., "af_heart" -> "a")
                lang_code = voice_name[0]
                found_lang_codes.add(lang_code)
        
        # Build language list for found codes
        languages = []
        for lang_code in sorted(found_lang_codes):
            if lang_code in language_map:
                lang_info = language_map[lang_code].copy()
                lang_info["code"] = lang_code
                languages.append(lang_info)
        
        return languages
        
    except Exception:
        # Fallback to basic English if repository access fails
        return [{"code": "a", "name": "American English", "iso": "en-US"}]

def validate_language_code(lang_code: str) -> str:
    """Validate and normalize language code"""
    if not lang_code:
        return "a"  # Default to American English
    
    lang_code = lang_code.lower().strip()
    language_map = get_language_map()
    
    if lang_code not in language_map:
        raise ValueError(f"Unsupported language code: {lang_code}. Supported codes: {', '.join(language_map.keys())}")
    
    return lang_code

@openai_router.post("/audio/speech", tags=["Audio"], response_model=None)
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
        # Validate language code
        lang_code = validate_language_code(request.language)
        
        # Parse voice specification
        voice_spec = parse_voice_specification(request.voice)
        
        # Set media type and filename
        media_type = get_media_type(request.response_format)
        
        if request.stream:
            if request.include_captions:
                # Streaming response with captions (JSON format)
                def generate_caption_chunks():
                    import json
                    import base64
                    
                    if isinstance(voice_spec, dict):
                        # Voice blending with streaming captions
                        for chunk in synthesize_streaming_with_voice_blend(
                            request.input, voice_spec, request.speed, request.response_format, lang_code, include_captions=True
                        ):
                            # Convert StreamingCaptionChunk to JSON-serializable format
                            word_data = [
                                {
                                    "text": wt.text,
                                    "start_time": wt.start_time,
                                    "end_time": wt.end_time,
                                    "phonemes": wt.phonemes
                                } for wt in chunk.word_timings
                            ]
                            
                            audio_data_b64 = base64.b64encode(chunk.audio_data).decode('utf-8') if chunk.audio_data else None
                            chunk_data = {
                                "word_timings": word_data,
                                "chunk_number": chunk.chunk_number,
                                "is_final": chunk.is_final,
                                "metadata": {
                                    "format": request.response_format,
                                    "speed": request.speed,
                                    "voice": request.voice,
                                    "language": request.language,
                                    "word_count": len(word_data),
                                    "audio_size_bytes": len(chunk.audio_data) if chunk.audio_data else 0,
                                    "audio_data": audio_data_b64,
                                    "generation_method": "streaming"
                                }
                            }
                            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                    else:
                        # Single voice streaming captions
                        for chunk in synthesize_streaming(
                            request.input, voice_spec, request.speed, request.response_format, lang_code, include_captions=True
                        ):
                            # Convert StreamingCaptionChunk to JSON-serializable format
                            word_data = [
                                {
                                    "text": wt.text,
                                    "start_time": wt.start_time,
                                    "end_time": wt.end_time,
                                    "phonemes": wt.phonemes
                                } for wt in chunk.word_timings
                            ]
                            
                            audio_data_b64 = base64.b64encode(chunk.audio_data).decode('utf-8') if chunk.audio_data else None
                            chunk_data = {
                                "word_timings": word_data,
                                "chunk_number": chunk.chunk_number,
                                "is_final": chunk.is_final,
                                "metadata": {
                                    "format": request.response_format,
                                    "speed": request.speed,
                                    "voice": request.voice,
                                    "language": request.language,
                                    "word_count": len(word_data),
                                    "audio_size_bytes": len(chunk.audio_data) if chunk.audio_data else 0,
                                    "audio_data": audio_data_b64,
                                    "generation_method": "streaming"
                                }
                            }
                            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                
                headers = {
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
                return StreamingResponse(generate_caption_chunks(), media_type="text/event-stream", headers=headers)
            else:
                # Streaming response (audio only)
                def generate_audio():
                    if isinstance(voice_spec, dict):
                        # Voice blending with streaming
                        for chunk in synthesize_streaming_with_voice_blend(
                            request.input, voice_spec, request.speed, request.response_format, lang_code, include_captions=False
                        ):
                            yield chunk
                    else:
                        # Single voice streaming
                        for chunk in synthesize_streaming(
                            request.input, voice_spec, request.speed, request.response_format, lang_code, include_captions=False
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
            if request.include_captions:
                # Generate audio with captions
                if isinstance(voice_spec, dict):
                    # Voice blending with captions
                    synthesis_result = synthesize_with_voice_blend(
                        request.input, voice_spec, request.speed, request.response_format, lang_code, include_captions=True
                    )
                else:
                    # Single voice with captions
                    synthesis_result = synthesize(
                        request.input, voice_spec, request.speed, request.response_format, lang_code, include_captions=True
                    )
                
                # Convert SynthesisResult to API response format
                word_timestamps = [
                    WordTimestamp(
                        text=wt.text,
                        start_time=wt.start_time,
                        end_time=wt.end_time,
                        phonemes=wt.phonemes
                    ) for wt in synthesis_result.word_timings
                ]
                
                # For read-along functionality, we need to include the audio data
                # Encode audio bytes as base64 for JSON response
                import base64
                audio_data_b64 = base64.b64encode(synthesis_result.audio_bytes).decode('utf-8')
                
                return CaptionedSpeechResponse(
                    audio_url=None,  # Direct audio data, no URL needed
                    captions=CaptionData(
                        words=word_timestamps,
                        total_duration=synthesis_result.total_duration,
                        sample_rate=synthesis_result.sample_rate
                    ),
                    metadata={
                        "format": request.response_format,
                        "speed": request.speed,
                        "voice": request.voice,
                        "language": request.language,
                        "word_count": len(word_timestamps),
                        "audio_size_bytes": len(synthesis_result.audio_bytes),
                        "audio_data": audio_data_b64,
                        "generation_method": "standard"
                    }
                )
            else:
                # Audio-only response
                if isinstance(voice_spec, dict):
                    # Voice blending
                    audio_bytes = synthesize_with_voice_blend(
                        request.input, voice_spec, request.speed, request.response_format, lang_code, include_captions=False
                    )
                else:
                    # Single voice
                    audio_bytes = synthesize(
                        request.input, voice_spec, request.speed, request.response_format, lang_code, include_captions=False
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

@openai_router.get("/audio/voices", response_model=OpenAIVoiceResponse, tags=["Audio"])
async def list_voices():
    """OpenAI compatible voice listing endpoint"""
    try:
        voices = get_supported_voices()
        return OpenAIVoiceResponse(voices=voices)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": f"Failed to list voices: {str(e)}", "type": "internal_server_error"}}
        )

@openai_router.get("/audio/languages", response_model=OpenAILanguageResponse, tags=["Audio"])
async def list_languages():
    """OpenAI compatible language listing endpoint"""
    try:
        languages = get_supported_languages()
        return OpenAILanguageResponse(languages=languages)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": f"Failed to list languages: {str(e)}", "type": "internal_server_error"}}
        )

# Debug and Monitoring Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint for monitoring"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=time.time() - startup_time,
        version="1.0.0"
    )

@app.get("/metrics", response_model=SystemMetricsResponse, tags=["Monitoring"])
async def system_metrics():
    """System metrics endpoint for monitoring"""
    # CPU and Memory metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # GPU metrics
    gpu_available = torch.cuda.is_available()
    gpu_memory_used_mb = None
    gpu_memory_total_mb = None
    
    if gpu_available:
        try:
            gpu_memory_used_mb = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_memory_total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        except Exception:
            pass
    
    return SystemMetricsResponse(
        cpu_percent=cpu_percent,
        memory_percent=memory.percent,
        memory_available_gb=memory.available / (1024**3),
        disk_usage_percent=disk.percent,
        gpu_available=gpu_available,
        gpu_memory_used_mb=gpu_memory_used_mb,
        gpu_memory_total_mb=gpu_memory_total_mb
    )

@app.get("/pipeline/status", response_model=PipelineStatusResponse, tags=["Debug"])
async def pipeline_status():
    """Pipeline status endpoint for debugging"""
    loaded_languages = list(_pipelines.keys())
    
    return PipelineStatusResponse(
        loaded_languages=loaded_languages,
        default_language="a",
        device=_device,
        pipeline_count=len(_pipelines),
        model_info={
            "model_name": "Kokoro-82M",
            "model_size": "82M parameters",
            "supported_languages": "9 languages"
        }
    )

@app.get("/debug", response_model=DebugInfoResponse, tags=["Debug"])
async def debug_info():
    """Comprehensive debug information endpoint"""
    # System info
    memory = psutil.virtual_memory()
    gpu_available = torch.cuda.is_available()
    
    system_info = {
        "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": memory.total / (1024**3),
        "gpu_available": gpu_available,
        "torch_version": torch.__version__,
        "device": _device,
        "uptime_hours": (time.time() - startup_time) / 3600
    }
    
    # Pipeline status
    pipeline_status = {
        "loaded_pipelines": list(_pipelines.keys()),
        "pipeline_count": len(_pipelines),
        "default_language": "a",
        "device": _device
    }
    
    # Performance info (placeholder for now)
    recent_performance = {
        "last_check": datetime.now().isoformat(),
        "avg_generation_time": "N/A",
        "requests_since_startup": "N/A"
    }
    
    return DebugInfoResponse(
        system_info=system_info,
        pipeline_status=pipeline_status,
        recent_performance=recent_performance
    )

# Include OpenAI router in main app
app.include_router(openai_router)