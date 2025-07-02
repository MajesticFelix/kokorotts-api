"""Configuration management for KokoroTTS API.

This module provides centralized configuration management with environment-based
settings for rate limiting, deployment modes, and other application settings.
"""

import os
from typing import Dict, Any, Optional
from enum import Enum


class DeploymentMode(str, Enum):
    """Deployment mode enumeration."""
    LOCAL = "local"
    CLOUD = "cloud"
    DEVELOPMENT = "development"


class RateLimitConfig:
    """Rate limiting configuration settings."""
    
    def __init__(self):
        self.enabled = self._get_bool_env("RATE_LIMITING_ENABLED", False)
        self.deployment_mode = DeploymentMode(
            os.getenv("DEPLOYMENT_MODE", DeploymentMode.LOCAL.value)
        )
        
        # Request rate limits
        self.requests_per_minute = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "30"))
        self.requests_per_hour = int(os.getenv("RATE_LIMIT_REQUESTS_PER_HOUR", "200"))
        self.requests_per_day = int(os.getenv("RATE_LIMIT_REQUESTS_PER_DAY", "1000"))
        
        # Character limits for TTS requests
        self.max_characters_per_request = int(os.getenv("RATE_LIMIT_MAX_CHARS_PER_REQUEST", "5000"))
        self.characters_per_hour = int(os.getenv("RATE_LIMIT_CHARS_PER_HOUR", "50000"))
        self.characters_per_day = int(os.getenv("RATE_LIMIT_CHARS_PER_DAY", "200000"))
        
        # Concurrent request limits
        self.max_concurrent_requests = int(os.getenv("RATE_LIMIT_MAX_CONCURRENT", "3"))
        
        # Redis configuration for distributed rate limiting
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.use_redis = self._get_bool_env("RATE_LIMIT_USE_REDIS", False)
        
        # Rate limit violation penalties
        self.violation_penalty_seconds = int(os.getenv("RATE_LIMIT_VIOLATION_PENALTY", "60"))
        self.max_violations_per_hour = int(os.getenv("RATE_LIMIT_MAX_VIOLATIONS_PER_HOUR", "10"))
    
    @staticmethod
    def _get_bool_env(key: str, default: bool = False) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")
    
    def is_enabled(self) -> bool:
        """Check if rate limiting is enabled."""
        return self.enabled
    
    def is_cloud_deployment(self) -> bool:
        """Check if running in cloud deployment mode."""
        return self.deployment_mode == DeploymentMode.CLOUD
    
    def get_limits_for_endpoint(self, endpoint_type: str) -> Dict[str, Any]:
        """Get rate limits for specific endpoint type."""
        if not self.enabled:
            return {}
        
        base_limits = {
            "requests_per_minute": self.requests_per_minute,
            "requests_per_hour": self.requests_per_hour,
            "requests_per_day": self.requests_per_day,
            "max_concurrent": self.max_concurrent_requests,
        }
        
        if endpoint_type == "speech":
            base_limits.update({
                "characters_per_request": self.max_characters_per_request,
                "characters_per_hour": self.characters_per_hour,
                "characters_per_day": self.characters_per_day,
            })
        
        return base_limits


class APIKeyConfig:
    """API key authentication configuration settings."""
    
    def __init__(self):
        # API key authentication settings
        self.enabled = self._get_bool_env("API_KEY_AUTHENTICATION", False)
        self.required = self._get_bool_env("API_KEY_REQUIRED", False)
        
        # Database configuration
        self.database_url = os.getenv("API_KEY_DATABASE_URL", "")
        
        # API key-specific rate limits (higher than IP-based limits)
        self.requests_per_minute = int(os.getenv("API_KEY_REQUESTS_PER_MINUTE", "100"))
        self.requests_per_hour = int(os.getenv("API_KEY_REQUESTS_PER_HOUR", "1000"))
        self.requests_per_day = int(os.getenv("API_KEY_REQUESTS_PER_DAY", "10000"))
        
        # Character limits for authenticated requests
        self.max_characters_per_request = int(os.getenv("API_KEY_MAX_CHARS_PER_REQUEST", "10000"))
        self.characters_per_hour = int(os.getenv("API_KEY_CHARS_PER_HOUR", "100000"))
        self.characters_per_day = int(os.getenv("API_KEY_CHARS_PER_DAY", "500000"))
        
        # Concurrent request limits for API keys
        self.max_concurrent_requests = int(os.getenv("API_KEY_MAX_CONCURRENT", "5"))
        
        # Admin access configuration
        self.admin_enabled = self._get_bool_env("API_KEY_ADMIN_ENABLED", True)
        self.admin_key = os.getenv("API_KEY_ADMIN_KEY", "")
    
    @staticmethod
    def _get_bool_env(key: str, default: bool = False) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")
    
    def is_enabled(self) -> bool:
        """Check if API key authentication is enabled."""
        return self.enabled
    
    def is_required(self) -> bool:
        """Check if API key authentication is required."""
        return self.required
    
    def is_admin_enabled(self) -> bool:
        """Check if admin endpoints are enabled."""
        return self.admin_enabled
    
    def get_database_url(self) -> str:
        """Get database URL for API keys, with fallback based on deployment mode."""
        if self.database_url:
            return self.database_url
        
        # Fallback based on deployment mode
        deployment_mode = os.getenv("DEPLOYMENT_MODE", "local")
        if deployment_mode == "local":
            return "sqlite:///./api_keys.db"
        else:
            return os.getenv("DATABASE_URL", "postgresql://kokoro:password@localhost:5432/kokorotts")
    
    def get_limits_for_tier(self, tier: str) -> Dict[str, Any]:
        """Get rate limits for a specific API key tier."""
        # Import here to avoid circular imports
        try:
            from .database.models import RateLimitTier
            tier_limits = RateLimitTier.get_limits(tier)
        except ImportError:
            # Fallback if database models not available
            tier_limits = {
                "requests_per_minute": self.requests_per_minute,
                "requests_per_hour": self.requests_per_hour,
                "requests_per_day": self.requests_per_day,
                "characters_per_request": self.max_characters_per_request,
                "characters_per_hour": self.characters_per_hour,
                "characters_per_day": self.characters_per_day,
                "max_concurrent_requests": self.max_concurrent_requests
            }
        
        # If tier has -1 (unlimited), use our configured API key limits as max
        limits = {}
        for key, value in tier_limits.items():
            if value == -1:
                # Use our default API key limits for "unlimited" tiers
                if key == "requests_per_minute":
                    limits[key] = self.requests_per_minute
                elif key == "requests_per_hour":
                    limits[key] = self.requests_per_hour
                elif key == "requests_per_day":
                    limits[key] = self.requests_per_day
                elif key == "characters_per_request":
                    limits[key] = self.max_characters_per_request
                elif key == "characters_per_hour":
                    limits[key] = self.characters_per_hour
                elif key == "characters_per_day":
                    limits[key] = self.characters_per_day
                elif key == "max_concurrent_requests":
                    limits[key] = self.max_concurrent_requests
                else:
                    limits[key] = -1  # Keep unlimited for unknown keys
            else:
                limits[key] = value
        
        return limits


class TTSConfig:
    """TTS-specific configuration settings."""
    
    def __init__(self):
        # Text processing
        max_text_env = os.getenv("TTS_MAX_TEXT_LENGTH", os.getenv("MAX_TEXT_LENGTH", "50000"))
        self.max_text_length = int(max_text_env) if max_text_env != "-1" else -1
        self.default_language = os.getenv("TTS_DEFAULT_LANGUAGE", os.getenv("DEFAULT_LANGUAGE", "a"))
        self.chunk_size = int(os.getenv("TTS_CHUNK_SIZE", "800"))
        self.max_chunk_size = int(os.getenv("TTS_MAX_CHUNK_SIZE", "1000"))
        
        # Audio processing
        self.sample_rate = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
        self.supported_formats = self._get_list_env("TTS_SUPPORTED_FORMATS", ["wav", "mp3", "flac", "ogg", "opus"])
        self.default_format = os.getenv("TTS_DEFAULT_FORMAT", "mp3")
        
        # Performance
        self.memory_limit_mb = int(os.getenv("TTS_MEMORY_LIMIT_MB", "1024"))
        self.batch_processing_threshold = int(os.getenv("TTS_BATCH_THRESHOLD", "10000"))
        self.enable_gpu = self._get_bool_env("TTS_ENABLE_GPU", True)
        
        # Voice settings
        self.voice_cache_duration = int(os.getenv("TTS_VOICE_CACHE_DURATION", os.getenv("VOICE_CACHE_DURATION", "3600")))
        self.enable_voice_blending = self._get_bool_env("TTS_ENABLE_VOICE_BLENDING", True)
        
        # Speed limits
        self.min_speed = float(os.getenv("TTS_MIN_SPEED", "0.25"))
        self.max_speed = float(os.getenv("TTS_MAX_SPEED", "4.0"))
        
        # Streaming settings
        self.streaming_chunk_size = int(os.getenv("TTS_STREAMING_CHUNK_SIZE", "600"))
        
        # Media type mappings
        self.media_types = {
            "wav": "audio/wav",
            "flac": "audio/flac", 
            "ogg": "audio/ogg",
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
        }
    
    @staticmethod
    def _get_bool_env(key: str, default: bool = False) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")
    
    @staticmethod
    def _get_list_env(key: str, default: list) -> list:
        """Get list environment variable (comma-separated)."""
        value = os.getenv(key)
        if value:
            return [item.strip() for item in value.split(",")]
        return default


class SecurityConfig:
    """Security and monitoring configuration."""
    
    def __init__(self):
        # CORS settings
        self.cors_origins = self._get_list_env("SECURITY_CORS_ORIGINS", self._get_list_env("CORS_ORIGINS", ["*"]))
        self.cors_credentials = self._get_bool_env("SECURITY_CORS_CREDENTIALS", True)
        self.cors_methods = self._get_list_env("SECURITY_CORS_METHODS", ["*"])
        self.cors_headers = self._get_list_env("SECURITY_CORS_HEADERS", ["*"])
        
        # Request handling
        self.trust_proxy_headers = self._get_bool_env("SECURITY_TRUST_PROXY_HEADERS", self._get_bool_env("TRUST_PROXY_HEADERS", True))
        self.max_request_size = int(os.getenv("SECURITY_MAX_REQUEST_SIZE", "10485760"))  # 10MB
        self.request_timeout = int(os.getenv("SECURITY_REQUEST_TIMEOUT", os.getenv("REQUEST_TIMEOUT", "300")))
        
        # Monitoring
        self.enable_metrics = self._get_bool_env("SECURITY_ENABLE_METRICS", True)
        self.enable_health_checks = self._get_bool_env("SECURITY_ENABLE_HEALTH_CHECKS", True)
        self.log_requests = self._get_bool_env("SECURITY_LOG_REQUESTS", False)
        
        # Admin access
        self.admin_ips = self._get_list_env("SECURITY_ADMIN_IPS", [])
        self.maintenance_mode = self._get_bool_env("SECURITY_MAINTENANCE_MODE", False)
    
    @staticmethod
    def _get_bool_env(key: str, default: bool = False) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")
    
    @staticmethod
    def _get_list_env(key: str, default: list) -> list:
        """Get list environment variable (comma-separated)."""
        value = os.getenv(key)
        if value:
            return [item.strip() for item in value.split(",")]
        return default


class AppConfig:
    """Main application configuration."""
    
    def __init__(self):
        # Component configurations
        self.rate_limit = RateLimitConfig()
        self.api_key = APIKeyConfig()
        self.tts = TTSConfig()
        self.security = SecurityConfig()
        
        # Application settings
        self.debug = self._get_bool_env("APP_DEBUG", self._get_bool_env("DEBUG", False))
        self.log_level = os.getenv("APP_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO"))
        self.host = os.getenv("APP_HOST", os.getenv("HOST", "0.0.0.0"))
        self.port = int(os.getenv("APP_PORT", os.getenv("PORT", "8000")))
        self.workers = int(os.getenv("APP_WORKERS", os.getenv("MAX_WORKERS", "4")))
        
        # Environment
        self.environment = os.getenv("APP_ENVIRONMENT", "development")
        self.deployment_mode = DeploymentMode(os.getenv("DEPLOYMENT_MODE", DeploymentMode.LOCAL.value))
        
        # Legacy compatibility properties
        self._setup_legacy_compatibility()
    
    def _setup_legacy_compatibility(self):
        """Provide backward compatibility for legacy configuration access."""
        # Legacy properties that map to new structure
        self.cors_origins = self.security.cors_origins
        self.trust_proxy_headers = self.security.trust_proxy_headers
        self.max_text_length = self.tts.max_text_length
        self.default_language = self.tts.default_language
        self.voice_cache_duration = self.tts.voice_cache_duration
        self.max_workers = self.workers
        self.request_timeout = self.security.request_timeout
    
    @staticmethod
    def _get_bool_env(key: str, default: bool = False) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")
    
    @staticmethod
    def _get_list_env(key: str, default: list) -> list:
        """Get list environment variable (comma-separated)."""
        value = os.getenv(key)
        if value:
            return [item.strip() for item in value.split(",")]
        return default
    
    def get_redis_config(self) -> Optional[Dict[str, Any]]:
        """Get Redis configuration for rate limiting."""
        if not self.rate_limit.use_redis:
            return None
        
        return {
            "url": self.rate_limit.redis_url,
            "decode_responses": True,
            "socket_timeout": 5,
            "socket_connect_timeout": 5,
            "retry_on_timeout": True,
            "health_check_interval": 30,
        }


# Global configuration instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get global configuration instance."""
    return config


def is_rate_limiting_enabled() -> bool:
    """Check if rate limiting is enabled."""
    return config.rate_limit.is_enabled()


def get_deployment_mode() -> DeploymentMode:
    """Get current deployment mode."""
    return config.rate_limit.deployment_mode


def get_client_ip(request, trust_proxy: bool = True) -> str:
    """Extract client IP address from request.
    
    Args:
        request: FastAPI request object
        trust_proxy: Whether to trust proxy headers
    
    Returns:
        Client IP address string
    """
    if trust_proxy:
        # Check common proxy headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain (client IP)
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        forwarded = request.headers.get("Forwarded")
        if forwarded:
            # Parse Forwarded header: for=192.0.2.43, for="[2001:db8:cafe::17]"
            import re
            match = re.search(r'for=([^;,\s]+)', forwarded)
            if match:
                ip = match.group(1).strip('"[]')
                return ip
    
    # Fallback to direct client IP
    client_host = getattr(request.client, "host", "unknown")
    return client_host if client_host != "unknown" else "127.0.0.1"


def format_rate_limit_error(limit_type: str, limit_value: str, reset_time: int) -> Dict[str, Any]:
    """Format standardized rate limit error response.
    
    Args:
        limit_type: Type of limit exceeded (requests, characters, concurrent)
        limit_value: The limit value that was exceeded
        reset_time: Seconds until limit resets
    
    Returns:
        Formatted error response dictionary
    """
    return {
        "error": {
            "message": f"Rate limit exceeded: {limit_type} limit of {limit_value}",
            "type": "rate_limit_exceeded",
            "limit_type": limit_type,
            "limit_value": limit_value,
            "retry_after": reset_time,
        }
    }


def is_api_key_authentication_enabled() -> bool:
    """Check if API key authentication is enabled."""
    return config.api_key.is_enabled()


def is_api_key_required() -> bool:
    """Check if API key authentication is required."""
    return config.api_key.is_required()


def is_api_key_admin_enabled() -> bool:
    """Check if API key admin endpoints are enabled."""
    return config.api_key.is_admin_enabled()