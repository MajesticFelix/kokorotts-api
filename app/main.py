"""FastAPI application for KokoroTTS text-to-speech API.

This module provides a REST API for text-to-speech synthesis with OpenAI-compatible
endpoints, voice blending, streaming responses, and word-level timing information.
"""

import base64
import io
import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import psutil
import torch
from fastapi import FastAPI, HTTPException, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import list_repo_files
from pydantic import BaseModel, Field

from .config import get_config, is_rate_limiting_enabled
from .tts_engine import (
    _device,
    _pipelines,
    synthesize,
    synthesize_streaming,
    synthesize_streaming_with_voice_blend,
    synthesize_with_voice_blend,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global startup time tracker
startup_time = time.time()

# Get configuration instance
config = get_config()

# Legacy constants for backward compatibility
SUPPORTED_FORMATS = config.tts.supported_formats
MIN_SPEED = config.tts.min_speed
MAX_SPEED = config.tts.max_speed

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
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.security.cors_origins,
    allow_credentials=config.security.cors_credentials,
    allow_methods=config.security.cors_methods,
    allow_headers=config.security.cors_headers,
)

# Set up rate limiting middleware (conditionally based on configuration)
rate_limiter = None
if is_rate_limiting_enabled():
    try:
        from .middleware.rate_limit import setup_rate_limiting
        rate_limiter, character_store, concurrent_store = setup_rate_limiting(app)
        logger.info(f"Rate limiting enabled for {config.rate_limit.deployment_mode} deployment")
    except ImportError as e:
        logger.warning(f"Rate limiting dependencies not available: {e}")
        logger.info("Rate limiting disabled - install 'slowapi' and 'redis' for rate limiting support")
else:
    logger.info("Rate limiting disabled by configuration")

# Set up API key authentication middleware (conditionally based on configuration)
api_key_middleware = None
try:
    from .middleware.api_key_auth import setup_api_key_authentication
    api_key_middleware = setup_api_key_authentication(app)
    if api_key_middleware:
        logger.info("API key authentication middleware enabled")
    else:
        logger.info("API key authentication middleware disabled")
except ImportError as e:
    logger.warning(f"API key authentication dependencies not available: {e}")
    logger.info("API key authentication disabled - install database dependencies")

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint providing API information and navigation."""
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
            "debug": "/debug",
        },
    }

@app.get("/test", response_class=HTMLResponse, tags=["Testing"])
async def test_page():
    """Serve the test page for trying out the TTS API."""
    try:
        with open("static/test.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Test page not found")

# OpenAI Compatible API Models
class OpenAISpeechRequest(BaseModel):
    """OpenAI-compatible speech synthesis request model."""
    
    model: str = Field(
        default="kokoro", 
        description="TTS model to use"
    )
    input: str = Field(
        ..., 
        description="Text to convert to speech (unlimited length)",
        min_length=1
    )
    voice: str = Field(
        default="af_heart", 
        description="Voice name or blended voices (e.g., 'af_bella+af_sky' or 'af_bella(2)+af_sky(1)')"
    )
    response_format: Optional[str] = Field(
        default="mp3", 
        description="Audio format: wav, mp3, flac, ogg, opus"
    )
    speed: Optional[float] = Field(
        default=1.0, 
        ge=MIN_SPEED, 
        le=MAX_SPEED, 
        description=f"Speech speed ({MIN_SPEED} to {MAX_SPEED})"
    )
    stream: Optional[bool] = Field(
        default=False, 
        description="Enable streaming response"
    )
    include_captions: Optional[bool] = Field(
        default=False, 
        description="Include per-word timing information in response"
    )
    language: Optional[str] = Field(
        default="a", 
        description="Language code (a=American English, b=British English, j=Japanese, z=Chinese, e=Spanish, f=French, h=Hindi, i=Italian, p=Portuguese)"
    )

class OpenAIVoiceResponse(BaseModel):
    """Response model for voice listing endpoint."""
    voices: List[str] = Field(..., description="List of available voice names")

class OpenAILanguageResponse(BaseModel):
    """Response model for language listing endpoint."""
    languages: List[Dict[str, str]] = Field(..., description="List of supported languages")

# Debug and Monitoring Response Models
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    version: str = Field(..., description="API version")

class APIKeyMetrics(BaseModel):
    enabled: bool = Field(..., description="Whether API key authentication is enabled")
    total_api_keys: int = Field(default=0, description="Total number of API keys")
    active_api_keys: int = Field(default=0, description="Number of active API keys")
    authenticated_requests: int = Field(default=0, description="Total authenticated requests")
    authentication_rate: float = Field(default=0.0, description="Percentage of authenticated requests")
    database_available: bool = Field(default=False, description="Whether database is available")

class RateLimitMetrics(BaseModel):
    enabled: bool = Field(..., description="Whether rate limiting is enabled")
    deployment_mode: str = Field(..., description="Current deployment mode")
    redis_connected: Optional[bool] = Field(None, description="Redis connection status")
    total_requests_blocked: int = Field(default=0, description="Total requests blocked since startup")
    active_rate_limited_ips: int = Field(default=0, description="Number of IPs currently rate limited")
    concurrent_requests: int = Field(default=0, description="Current concurrent requests")
    character_usage_stats: Dict[str, int] = Field(default_factory=dict, description="Character usage statistics")

class SystemMetricsResponse(BaseModel):
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    gpu_available: bool
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    rate_limiting: Optional[RateLimitMetrics] = None
    api_key_auth: Optional[APIKeyMetrics] = None

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

# Rate Limiting Admin Models
class RateLimitUsageStats(BaseModel):
    time_window_hours: int
    total_requests: int
    blocked_requests: int
    unique_ips: int
    total_characters: int
    requests_per_hour: float
    characters_per_hour: float
    block_rate: float
    top_endpoints: List[Dict[str, Any]]
    top_ips: List[Dict[str, Any]]

class RateLimitViolationReport(BaseModel):
    time_window_hours: int
    total_violations: int
    unique_violating_ips: int
    violations_by_reason: Dict[str, int]
    violations_by_hour: List[Dict[str, Any]]
    top_violating_ips: List[Dict[str, Any]]

class IPDetailsResponse(BaseModel):
    ip: str
    total_requests: int
    blocked_requests: int
    success_rate: float
    total_characters: int
    violation_count: int
    first_seen: Optional[str]
    last_seen: Optional[str]
    avg_request_interval_seconds: float
    recent_violations: List[Dict[str, Any]]

class AbusePattern(BaseModel):
    type: str
    ip: str
    severity: str
    details: Dict[str, Any] = Field(default_factory=dict)

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
    """Parse OpenAI voice specification into voice name or voice weights dict.
    
    Args:
        voice_spec: Voice specification string
        
    Returns:
        Single voice name or dictionary of voice weights
        
    Examples:
        - 'af_bella' -> 'af_bella'
        - 'af_bella+af_sky' -> {'af_bella': 0.5, 'af_sky': 0.5}
        - 'af_bella(2)+af_sky(1)' -> {'af_bella': 0.67, 'af_sky': 0.33}
        
    Raises:
        ValueError: If voice specification format is invalid
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
    """Get proper media type for audio format.
    
    Args:
        format_name: Audio format name
        
    Returns:
        MIME type string for the format
    """
    return config.tts.media_types.get(format_name.lower(), "audio/mpeg")

# Cache for voice list to avoid repeated API calls
_voice_cache: Optional[List[str]] = None
_voice_cache_time: Optional[float] = None

def get_supported_voices() -> List[str]:
    """Get list of supported voices from repository.
    
    Returns:
        List of available voice names
        
    Raises:
        RuntimeError: If unable to fetch voice list
    """
    global _voice_cache, _voice_cache_time
    
    # Use cached result if available and not expired
    current_time = time.time()
    if (
        _voice_cache is not None
        and _voice_cache_time is not None
        and current_time - _voice_cache_time < config.tts.voice_cache_duration
    ):
        return _voice_cache
    
    try:
        files = list_repo_files(repo_id="hexgrad/Kokoro-82M")
        voice_files = [f for f in files if f.endswith(".pt") and "voices" in f]
        voices = [f.split("/")[-1].replace(".pt", "") for f in voice_files]
        
        # Cache the result
        _voice_cache = voices
        _voice_cache_time = current_time
        
        return voices
    except Exception as e:
        logger.error(f"Failed to fetch voice list: {e}")
        # Return cached result if available, otherwise raise
        if _voice_cache is not None:
            logger.warning("Using cached voice list due to fetch failure")
            return _voice_cache
        raise RuntimeError(f"Unable to fetch voice list: {e}") from e

def get_language_map() -> Dict[str, Dict[str, str]]:
    """Get the mapping of language codes to language information.
    
    Returns:
        Dictionary mapping language codes to language metadata
    """
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
    """Infer supported languages from actual voice files in repository.
    
    Returns:
        List of language information dictionaries
    """
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
    """Validate and normalize language code.
    
    Args:
        lang_code: Input language code
        
    Returns:
        Validated and normalized language code
        
    Raises:
        ValueError: If language code is not supported
    """
    if not lang_code:
        return config.tts.default_language
    
    lang_code = lang_code.lower().strip()
    language_map = get_language_map()
    
    if lang_code not in language_map:
        raise ValueError(f"Unsupported language code: {lang_code}. Supported codes: {', '.join(language_map.keys())}")
    
    return lang_code

def validate_speech_request(request: OpenAISpeechRequest) -> None:
    """Centralized validation for speech synthesis requests.
    
    Args:
        request: Speech synthesis request to validate
        
    Raises:
        HTTPException: For validation errors
    """
    # Validate format
    if request.response_format not in config.tts.supported_formats:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Unsupported response_format: {request.response_format}. Supported: {', '.join(config.tts.supported_formats)}",
                    "type": "invalid_request_error",
                }
            },
        )
    
    # Validate input text
    if not request.input or not request.input.strip():
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "Input text cannot be empty", "type": "invalid_request_error"}}
        )
    
    # Validate text length (both global and rate limiting)
    input_length = len(request.input)
    
    # Global maximum text length check (skip if unlimited)
    max_global_length = config.max_text_length
    if max_global_length > 0 and input_length > max_global_length:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Input text too long: {input_length} characters. Maximum allowed: {max_global_length}",
                    "type": "invalid_request_error",
                }
            },
        )
    
    # Rate limiting character check (if enabled)
    if is_rate_limiting_enabled():
        max_chars_per_request = config.rate_limit.max_characters_per_request
        if input_length > max_chars_per_request:
            from .config import format_rate_limit_error
            raise HTTPException(
                status_code=429,
                detail=format_rate_limit_error(
                    "characters_per_request",
                    str(max_chars_per_request),
                    0
                )
            )
    
    # Validate speed parameter
    if not (config.tts.min_speed <= request.speed <= config.tts.max_speed):
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Speed must be between {config.tts.min_speed} and {config.tts.max_speed}",
                    "type": "invalid_request_error",
                }
            },
        )

def apply_rate_limiting_decorator(func):
    """Apply rate limiting decorator if rate limiting is enabled.
    
    NOTE: Disabled for TTS endpoints - our middleware provides better rate limiting
    with character limits, concurrent limits, and API key support.
    """
    if rate_limiter and is_rate_limiting_enabled():
        # NOTE: Commented out slowapi decorators due to compatibility issues with FastAPI Pydantic models
        # Our RateLimitMiddleware provides superior rate limiting anyway
        logger.info("Rate limiting handled by middleware (slowapi decorators disabled for compatibility)")
        
        # Future: Could add slowapi decorators for endpoints that use raw Request objects
        # config = get_config()
        # func = rate_limiter.limit(f"{config.rate_limit.requests_per_minute}/minute")(func)
        # func = rate_limiter.limit(f"{config.rate_limit.requests_per_hour}/hour")(func)
        # func = rate_limiter.limit(f"{config.rate_limit.requests_per_day}/day")(func)
        
    return func

def apply_light_rate_limiting_decorator(func):
    """Apply lighter rate limiting for informational endpoints.
    
    NOTE: Disabled to avoid compatibility issues. Middleware provides rate limiting.
    """
    if rate_limiter and is_rate_limiting_enabled():
        # NOTE: Commented out slowapi decorators - middleware handles rate limiting
        logger.info("Rate limiting handled by middleware (slowapi decorators disabled for compatibility)")
        
        # Future: Could add decorators for endpoints that need them and are compatible
        # func = rate_limiter.limit("60/minute")(func)  # Higher limit for info endpoints
        # func = rate_limiter.limit("500/hour")(func)
        
    return func

@openai_router.post("/audio/speech", tags=["Audio"], response_model=None)
@apply_rate_limiting_decorator
async def create_speech(request: OpenAISpeechRequest):
    """OpenAI compatible speech synthesis endpoint.
    
    Generates speech from text using KokoroTTS with support for:
    - Multiple voices and voice blending
    - Streaming and non-streaming responses  
    - Word-level timing captions
    - Multiple audio formats
    
    Args:
        request: Speech synthesis request parameters
        
    Returns:
        Audio data (streaming or complete) or captioned response
        
    Raises:
        HTTPException: For validation errors or synthesis failures
    """
    # Centralized validation
    validate_speech_request(request)
    
    try:
        # Log request details
        logger.info(
            f"Speech synthesis request: voice={request.voice}, format={request.response_format}, "
            f"speed={request.speed}, stream={request.stream}, captions={request.include_captions}, "
            f"lang={request.language}, text_length={len(request.input)}"
        )
        
        # Validate language code
        lang_code = validate_language_code(request.language)
        
        # Parse voice specification
        voice_spec = parse_voice_specification(request.voice)
        
        # Set media type and filename
        media_type = get_media_type(request.response_format)
        
        def create_streaming_response():
            """Create appropriate streaming response based on request parameters."""
            if isinstance(voice_spec, dict):
                # Voice blending streaming
                stream_generator = synthesize_streaming_with_voice_blend(
                    request.input, voice_spec, request.speed, request.response_format, lang_code, include_captions=request.include_captions
                )
            else:
                # Single voice streaming
                stream_generator = synthesize_streaming(
                    request.input, voice_spec, request.speed, request.response_format, lang_code, include_captions=request.include_captions
                )
            
            if request.include_captions:
                # Streaming response with captions (JSON format)
                def generate_caption_chunks():
                    for chunk in stream_generator:
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
                
                return StreamingResponse(
                    generate_caption_chunks(),
                    media_type="text/event-stream",
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive"
                    }
                )
            else:
                # Streaming response (audio only)
                return StreamingResponse(
                    stream_generator,
                    media_type=media_type,
                    headers={
                        "Content-Type": media_type,
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive"
                    }
                )
        
        if request.stream:
            return create_streaming_response()
        else:
            # Non-streaming response - use unified synthesis function
            if isinstance(voice_spec, dict):
                # Voice blending
                synthesis_result = synthesize_with_voice_blend(
                    request.input, voice_spec, request.speed, request.response_format, lang_code, include_captions=request.include_captions
                )
            else:
                # Single voice
                synthesis_result = synthesize(
                    request.input, voice_spec, request.speed, request.response_format, lang_code, include_captions=request.include_captions
                )
            
            if request.include_captions:
                # Convert SynthesisResult to API response format
                word_timestamps = [
                    WordTimestamp(
                        text=wt.text,
                        start_time=wt.start_time,
                        end_time=wt.end_time,
                        phonemes=wt.phonemes
                    ) for wt in synthesis_result.word_timings
                ]
                
                # Encode audio bytes as base64 for JSON response
                audio_data_b64 = base64.b64encode(synthesis_result.audio_bytes).decode("utf-8")
                
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
                        "generation_method": "standard",
                        "total_characters": len(request.input)
                    }
                )
            else:
                # Audio-only response
                audio_stream = io.BytesIO(synthesis_result)
                filename = f"speech.{request.response_format}"
                headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
                logger.info(f"Generated audio: {len(synthesis_result)} bytes, format={request.response_format}")
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
@apply_light_rate_limiting_decorator
async def list_voices(request: Request):
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
@apply_light_rate_limiting_decorator  
async def list_languages(request: Request):
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
    
    # Rate limiting metrics
    rate_limit_metrics = None
    if is_rate_limiting_enabled():
        try:
            from .middleware.rate_limit import get_rate_limit_metrics
            rate_limit_metrics = await get_rate_limit_metrics()
        except Exception as e:
            logger.warning(f"Failed to get rate limit metrics: {e}")
            rate_limit_metrics = RateLimitMetrics(
                enabled=True,
                deployment_mode=config.rate_limit.deployment_mode.value,
                redis_connected=None,
                total_requests_blocked=0,
                active_rate_limited_ips=0,
                concurrent_requests=0,
                character_usage_stats={}
            )
    else:
        rate_limit_metrics = RateLimitMetrics(
            enabled=False,
            deployment_mode=config.rate_limit.deployment_mode.value,
            redis_connected=None,
            total_requests_blocked=0,
            active_rate_limited_ips=0,
            concurrent_requests=0,
            character_usage_stats={}
        )
    
    # API key authentication metrics
    api_key_metrics = None
    try:
        from .config import is_api_key_authentication_enabled
        from .database.connection import is_database_available
        
        api_key_enabled = is_api_key_authentication_enabled()
        db_available = is_database_available()
        
        if api_key_enabled and db_available:
            # Get API key statistics
            from .utils.key_manager import KeyManager
            
            try:
                api_keys, total_keys = KeyManager.list_api_keys(None, 1, 1000)
                active_keys = sum(1 for key in api_keys if key.is_active)
                
                # Get middleware stats if available
                auth_stats = {"authenticated_requests": 0, "total_requests": 1}
                if api_key_middleware:
                    auth_stats = api_key_middleware.get_stats()
                
                api_key_metrics = APIKeyMetrics(
                    enabled=True,
                    total_api_keys=total_keys,
                    active_api_keys=active_keys,
                    authenticated_requests=auth_stats.get("authenticated_requests", 0),
                    authentication_rate=auth_stats.get("authentication_rate", 0.0),
                    database_available=True
                )
            except Exception as e:
                logger.debug(f"Error getting API key metrics: {e}")
                api_key_metrics = APIKeyMetrics(
                    enabled=True,
                    database_available=False
                )
        else:
            api_key_metrics = APIKeyMetrics(
                enabled=api_key_enabled,
                database_available=db_available
            )
    except ImportError:
        api_key_metrics = APIKeyMetrics(enabled=False, database_available=False)
    
    return SystemMetricsResponse(
        cpu_percent=cpu_percent,
        memory_percent=memory.percent,
        memory_available_gb=memory.available / (1024**3),
        disk_usage_percent=disk.percent,
        gpu_available=gpu_available,
        gpu_memory_used_mb=gpu_memory_used_mb,
        gpu_memory_total_mb=gpu_memory_total_mb,
        rate_limiting=rate_limit_metrics,
        api_key_auth=api_key_metrics
    )

@app.get("/pipeline/status", response_model=PipelineStatusResponse, tags=["Debug"])
async def pipeline_status():
    """Pipeline status endpoint for debugging"""
    loaded_languages = list(_pipelines.keys())
    
    return PipelineStatusResponse(
        loaded_languages=loaded_languages,
        default_language=config.tts.default_language,
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
        "default_language": config.tts.default_language,
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

# Rate Limiting Admin Endpoints (only available when rate limiting is enabled)
@app.get("/admin/rate-limits/usage", response_model=RateLimitUsageStats, tags=["Admin"])
async def get_rate_limit_usage_stats(hours: int = 24):
    """Get rate limiting usage statistics for admin monitoring."""
    if not is_rate_limiting_enabled():
        raise HTTPException(
            status_code=404,
            detail={"error": {"message": "Rate limiting is not enabled", "type": "feature_disabled"}}
        )
    
    try:
        from .utils.rate_limit_monitor import get_monitor
        monitor = get_monitor()
        stats = monitor.get_usage_stats(hours)
        return RateLimitUsageStats(**stats)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": f"Failed to get usage stats: {str(e)}", "type": "internal_server_error"}}
        )

@app.get("/admin/rate-limits/violations", response_model=RateLimitViolationReport, tags=["Admin"])
async def get_rate_limit_violations(hours: int = 24):
    """Get rate limiting violation report for admin monitoring."""
    if not is_rate_limiting_enabled():
        raise HTTPException(
            status_code=404,
            detail={"error": {"message": "Rate limiting is not enabled", "type": "feature_disabled"}}
        )
    
    try:
        from .utils.rate_limit_monitor import get_monitor
        monitor = get_monitor()
        violations = monitor.get_violation_report(hours)
        return RateLimitViolationReport(**violations)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": f"Failed to get violation report: {str(e)}", "type": "internal_server_error"}}
        )

@app.get("/admin/rate-limits/ip/{ip_address}", response_model=IPDetailsResponse, tags=["Admin"])
async def get_ip_details(ip_address: str):
    """Get detailed information about a specific IP address."""
    if not is_rate_limiting_enabled():
        raise HTTPException(
            status_code=404,
            detail={"error": {"message": "Rate limiting is not enabled", "type": "feature_disabled"}}
        )
    
    try:
        from .utils.rate_limit_monitor import get_monitor
        monitor = get_monitor()
        details = monitor.get_ip_details(ip_address)
        
        if "error" in details:
            raise HTTPException(
                status_code=404,
                detail={"error": {"message": details["error"], "type": "not_found"}}
            )
        
        return IPDetailsResponse(**details)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": f"Failed to get IP details: {str(e)}", "type": "internal_server_error"}}
        )

@app.get("/admin/rate-limits/abuse-patterns", response_model=List[AbusePattern], tags=["Admin"])
async def get_abuse_patterns():
    """Detect and return potential abuse patterns."""
    if not is_rate_limiting_enabled():
        raise HTTPException(
            status_code=404,
            detail={"error": {"message": "Rate limiting is not enabled", "type": "feature_disabled"}}
        )
    
    try:
        from .utils.rate_limit_monitor import get_monitor
        monitor = get_monitor()
        patterns = monitor.detect_abuse_patterns()
        
        # Convert to response models
        abuse_patterns = []
        for pattern in patterns:
            # Extract details for the response
            details = {k: v for k, v in pattern.items() if k not in ["type", "ip", "severity"]}
            abuse_patterns.append(AbusePattern(
                type=pattern["type"],
                ip=pattern["ip"],
                severity=pattern["severity"],
                details=details
            ))
        
        return abuse_patterns
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": f"Failed to detect abuse patterns: {str(e)}", "type": "internal_server_error"}}
        )

@app.get("/admin/rate-limits/export", tags=["Admin"])
async def export_rate_limit_data(format: str = "json"):
    """Export rate limiting monitoring data."""
    if not is_rate_limiting_enabled():
        raise HTTPException(
            status_code=404,
            detail={"error": {"message": "Rate limiting is not enabled", "type": "feature_disabled"}}
        )
    
    if format.lower() not in ["json", "csv"]:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "Supported formats: json, csv", "type": "invalid_request_error"}}
        )
    
    try:
        from .utils.rate_limit_monitor import get_monitor
        monitor = get_monitor()
        export_data = monitor.export_metrics(format)
        
        # Set appropriate content type and filename
        if format.lower() == "json":
            media_type = "application/json"
            filename = f"rate_limit_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:  # csv
            media_type = "text/csv"
            filename = f"rate_limit_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        from fastapi.responses import Response
        return Response(
            content=export_data,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": f"Failed to export data: {str(e)}", "type": "internal_server_error"}}
        )

@app.get("/admin/rate-limits/config", tags=["Admin"])
async def get_rate_limit_config():
    """Get current rate limiting configuration."""
    if not is_rate_limiting_enabled():
        return {"rate_limiting_enabled": False, "deployment_mode": config.rate_limit.deployment_mode.value}
    
    return {
        "rate_limiting_enabled": True,
        "deployment_mode": config.rate_limit.deployment_mode.value,
        "limits": {
            "requests_per_minute": config.rate_limit.requests_per_minute,
            "requests_per_hour": config.rate_limit.requests_per_hour,
            "requests_per_day": config.rate_limit.requests_per_day,
            "characters_per_request": config.rate_limit.max_characters_per_request,
            "characters_per_hour": config.rate_limit.characters_per_hour,
            "characters_per_day": config.rate_limit.characters_per_day,
            "max_concurrent_requests": config.rate_limit.max_concurrent_requests
        },
        "redis_config": {
            "use_redis": config.rate_limit.use_redis,
            "redis_url": config.rate_limit.redis_url if config.rate_limit.use_redis else None
        },
        "penalties": {
            "violation_penalty_seconds": config.rate_limit.violation_penalty_seconds,
            "max_violations_per_hour": config.rate_limit.max_violations_per_hour
        }
    }

# Test endpoint with direct slowapi decorator
if rate_limiter:
    @app.get("/test-rate-limit")
    @rate_limiter.limit("2/minute")
    async def test_rate_limit(request: Request):
        """Test endpoint to verify slowapi functionality."""
        return {"message": "Rate limit test", "timestamp": time.time()}

# Include OpenAI router in main app
app.include_router(openai_router)

# Include Admin router for API key management (conditionally)
try:
    from .routers.admin import admin_router
    from .config import is_api_key_admin_enabled
    
    if is_api_key_admin_enabled():
        app.include_router(admin_router)
        logger.info("Admin API endpoints enabled")
    else:
        logger.info("Admin API endpoints disabled by configuration")
except ImportError as e:
    logger.warning(f"Admin router not available: {e}")
    logger.info("Admin API endpoints disabled - install database dependencies")