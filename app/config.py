"""Configuration management for KokoroTTS API using environment variables.

This module provides centralized configuration management using Pydantic Settings
to load and validate configuration from environment variables with sensible defaults.
"""

from typing import List, Optional, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TTSEngineConfig(BaseSettings):
    """Configuration for the TTS engine and audio processing."""
    
    # Device configuration
    device: str = Field(default="auto", description="Device to use: auto, cpu, or cuda")
    
    # Audio processing settings
    sample_rate: int = Field(default=24000, description="Audio sample rate in Hz")
    default_chunk_size: int = Field(default=800, description="Default text chunk size for processing")
    max_chunk_size: int = Field(default=1000, description="Maximum text chunk size")
    streaming_chunk_size: int = Field(default=800, description="Chunk size for streaming responses")
    
    # Audio format settings
    mp3_bitrate: str = Field(default="192k", description="MP3 encoding bitrate")
    opus_codec: str = Field(default="libopus", description="Opus codec to use")
    
    # Memory and batch processing
    memory_limit_mb: float = Field(default=1024, description="Memory limit for audio processing in MB")
    batch_size: int = Field(default=5, description="Number of chunks to process in parallel")
    
    # Model and cache settings
    cache_dir: str = Field(default="/app/cache", description="Cache directory for models and data")
    model_dir: str = Field(default="/app/models", description="Model storage directory")
    repo_id: str = Field(default="hexgrad/Kokoro-82M", description="HuggingFace model repository ID")
    
    model_config = SettingsConfigDict(
        env_prefix="KOKORO_",
        case_sensitive=False
    )


class APIConfig(BaseSettings):
    """Configuration for the FastAPI application and API behavior."""
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Host to bind the server to")
    port: int = Field(default=8000, description="Port to bind the server to")
    workers: int = Field(default=1, description="Number of worker processes")
    log_level: str = Field(default="info", description="Logging level")
    debug: bool = Field(default=False, description="Enable debug mode")
    reload: bool = Field(default=False, description="Enable auto-reload for development")
    
    # API metadata
    title: str = Field(default="Kokoro TTS API", description="API title")
    version: str = Field(default="1.0.0", description="API version")
    description: str = Field(
        default="OpenAI-compatible TTS API using Kokoro model with voice blending support",
        description="API description"
    )
    
    # API behavior settings
    supported_formats: Union[List[str], str] = Field(
        default=["wav", "mp3", "flac", "ogg", "opus"],
        description="Supported audio formats"
    )
    min_speed: float = Field(default=0.25, description="Minimum speech speed")
    max_speed: float = Field(default=4.0, description="Maximum speech speed")
    default_language: str = Field(default="a", description="Default language code")
    default_voice: str = Field(default="af_heart", description="Default voice name")
    default_format: str = Field(default="mp3", description="Default audio format")
    voice_cache_duration: int = Field(default=3600, description="Voice cache duration in seconds")
    
    # CORS settings
    cors_origins: Union[List[str], str] = Field(
        default=["*"], 
        description="CORS allowed origins"
    )
    cors_methods: Union[List[str], str] = Field(
        default=["*"], 
        description="CORS allowed methods"
    )
    cors_headers: Union[List[str], str] = Field(
        default=["*"], 
        description="CORS allowed headers"
    )
    allow_credentials: bool = Field(default=True, description="Allow CORS credentials")
    
    # Static files
    static_directory: str = Field(default="static", description="Static files directory")
    docs_url: str = Field(default="/docs", description="Swagger UI URL")
    redoc_url: str = Field(default="/redoc", description="ReDoc URL")
    
    @field_validator('supported_formats', mode='before')
    @classmethod
    def parse_supported_formats(cls, v):
        if isinstance(v, str):
            if not v.strip():  # Handle empty strings
                return ["wav", "mp3", "flac", "ogg", "opus"]
            formats = [format.strip() for format in v.split(',') if format.strip()]
            # Validate formats
            valid_formats = ['wav', 'mp3', 'flac', 'ogg', 'opus']
            invalid_formats = [f for f in formats if f not in valid_formats]
            if invalid_formats:
                raise ValueError(f'Invalid audio formats: {invalid_formats}. Valid formats: {valid_formats}')
            return formats
        return v
    
    @field_validator('cors_origins', 'cors_methods', 'cors_headers', mode='before')
    @classmethod
    def parse_cors_lists(cls, v):
        if isinstance(v, str):
            if not v.strip():  # Handle empty strings
                return ["*"]
            return [item.strip() for item in v.split(',') if item.strip()]
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['debug', 'info', 'warning', 'error', 'critical']
        if v.lower() not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v.lower()
    
    @field_validator('min_speed', 'max_speed')
    @classmethod
    def validate_speed_range(cls, v):
        if v <= 0:
            raise ValueError('Speed values must be positive')
        if v < 0.1 or v > 10.0:
            raise ValueError('Speed values must be between 0.1 and 10.0')
        return v
    
    model_config = SettingsConfigDict(
        env_prefix="API_",
        case_sensitive=False,
        env_parse_none_str=""
    )


class SecurityConfig(BaseSettings):
    """Configuration for API security features."""
    
    # API Key authentication
    api_key_enabled: bool = Field(default=False, description="Enable API key authentication")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    api_key_header: str = Field(default="X-API-Key", description="Header name for API key")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=False, description="Enable rate limiting")
    rate_limit_per_minute: int = Field(default=60, description="Requests per minute limit")
    rate_limit_burst: int = Field(default=10, description="Burst limit for rate limiting")
    
    model_config = SettingsConfigDict(
        env_prefix="SECURITY_",
        case_sensitive=False
    )


class MonitoringConfig(BaseSettings):
    """Configuration for monitoring and observability."""
    
    # Health checks
    health_check_enabled: bool = Field(default=True, description="Enable health check endpoint")
    health_check_path: str = Field(default="/health", description="Health check endpoint path")
    
    # Metrics
    metrics_enabled: bool = Field(default=True, description="Enable metrics endpoint")
    metrics_port: Optional[int] = Field(default=None, description="Separate port for metrics")
    metrics_path: str = Field(default="/metrics", description="Metrics endpoint path")
    
    # Logging
    log_format: str = Field(default="json", description="Log format: json or text")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    # External monitoring
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN for error tracking")
    sentry_environment: str = Field(default="production", description="Sentry environment")
    
    model_config = SettingsConfigDict(
        env_prefix="MONITORING_",
        case_sensitive=False
    )


class Settings(BaseSettings):
    """Main configuration class combining all config sections."""
    
    # Environment
    environment: str = Field(default="production", description="Application environment")
    
    # Sub-configurations
    tts: TTSEngineConfig = TTSEngineConfig()
    api: APIConfig = APIConfig()
    security: SecurityConfig = SecurityConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize sub-configurations with environment variables
        self.tts = TTSEngineConfig()
        self.api = APIConfig()
        self.security = SecurityConfig()
        self.monitoring = MonitoringConfig()
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        valid_environments = ['development', 'testing', 'staging', 'production']
        if v.lower() not in valid_environments:
            raise ValueError(f'environment must be one of {valid_environments}')
        return v.lower()
    
    def model_post_init(self, __context=None) -> None:
        """Post-initialization validation and setup."""
        # Validate speed range consistency
        if hasattr(self.api, 'min_speed') and hasattr(self.api, 'max_speed'):
            if self.api.min_speed >= self.api.max_speed:
                raise ValueError('min_speed must be less than max_speed')
        
        # Validate chunk size consistency
        if hasattr(self.tts, 'default_chunk_size') and hasattr(self.tts, 'max_chunk_size'):
            if self.tts.default_chunk_size > self.tts.max_chunk_size:
                raise ValueError('default_chunk_size must not exceed max_chunk_size')
        
        # Validate memory and batch size consistency
        if hasattr(self.tts, 'memory_limit_mb') and self.tts.memory_limit_mb < 256:
            raise ValueError('memory_limit_mb should be at least 256 MB for stable operation')
        
        # Validate device setting
        if hasattr(self.tts, 'device'):
            valid_devices = ['auto', 'cpu', 'cuda']
            if self.tts.device not in valid_devices:
                raise ValueError(f'device must be one of {valid_devices}')
    
    def get_media_types(self) -> dict:
        """Get media type mapping for supported formats."""
        return {
            "wav": "audio/wav",
            "flac": "audio/flac",
            "ogg": "audio/ogg",
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
        }
    
    def get_audio_format_config(self) -> dict:
        """Get audio format configuration for encoding."""
        return {
            "soundfile_formats": ["wav", "flac", "ogg"],
            "pydub_formats": ["mp3", "opus"],
            "format_configs": {
                "mp3": {"format": "mp3", "bitrate": self.tts.mp3_bitrate},
                "opus": {"format": "opus", "codec": self.tts.opus_codec}
            }
        }
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


# Global settings instance
try:
    settings = Settings()
except Exception as e:
    import logging
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to load initial configuration: {e}")
    logger.error("Please check your environment variables and .env file")
    raise RuntimeError(f"Configuration initialization failed: {e}") from e


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment variables."""
    global settings
    try:
        settings = Settings()
        return settings
    except Exception as e:
        # Log the error and provide helpful message
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to reload configuration: {e}")
        raise RuntimeError(f"Configuration validation failed: {e}") from e

def validate_configuration() -> bool:
    """Validate the current configuration and return True if valid.
    
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    try:
        settings = get_settings()
        
        # Additional runtime validations
        import os
        
        # Check if cache directories are accessible
        cache_dir = settings.tts.cache_dir
        model_dir = settings.tts.model_dir
        
        try:
            os.makedirs(cache_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)
        except PermissionError:
            raise ValueError(f"Cannot create or access cache directories: {cache_dir}, {model_dir}")
        
        # Validate API key configuration
        if settings.security.api_key_enabled and not settings.security.api_key:
            raise ValueError("API key is enabled but no key is provided")
        
        # Validate rate limiting configuration
        if settings.security.rate_limit_enabled:
            if settings.security.rate_limit_per_minute <= 0:
                raise ValueError("rate_limit_per_minute must be positive when rate limiting is enabled")
            if settings.security.rate_limit_burst <= 0:
                raise ValueError("rate_limit_burst must be positive when rate limiting is enabled")
        
        return True
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Configuration validation failed: {e}")
        raise