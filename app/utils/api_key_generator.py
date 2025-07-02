"""API key generation and management utilities."""

import secrets
import string
import hashlib
import re
from typing import Optional, Tuple
import bcrypt
import logging

logger = logging.getLogger(__name__)


class APIKeyGenerator:
    """Utility class for generating and managing API keys."""
    
    # API key format: kk_{project_name}_{random_string}
    PREFIX = "kk_"
    RANDOM_LENGTH = 32
    VALID_PROJECT_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
    
    @classmethod
    def generate_api_key(cls, project_name: str) -> str:
        """Generate a new API key for a project.
        
        Args:
            project_name: Name of the project (will be sanitized)
            
        Returns:
            Generated API key in format: kk_{project_name}_{random_string}
            
        Raises:
            ValueError: If project name is invalid
        """
        # Sanitize and validate project name
        sanitized_name = cls._sanitize_project_name(project_name)
        if not sanitized_name:
            raise ValueError(f"Invalid project name: {project_name}")
        
        # Generate random component
        random_part = cls._generate_random_string(cls.RANDOM_LENGTH)
        
        # Combine components
        api_key = f"{cls.PREFIX}{sanitized_name}_{random_part}"
        
        logger.info(f"Generated API key for project: {sanitized_name}")
        return api_key
    
    @classmethod
    def _sanitize_project_name(cls, project_name: str) -> str:
        """Sanitize project name for use in API key.
        
        Args:
            project_name: Raw project name
            
        Returns:
            Sanitized project name safe for API key use
        """
        if not project_name:
            return ""
        
        # Remove whitespace and convert to lowercase
        sanitized = project_name.strip().lower()
        
        # Replace spaces and invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', sanitized)
        
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Limit length
        sanitized = sanitized[:20]
        
        # Validate final result
        if not cls.VALID_PROJECT_NAME_PATTERN.match(sanitized):
            return ""
        
        return sanitized
    
    @classmethod
    def _generate_random_string(cls, length: int) -> str:
        """Generate cryptographically secure random string.
        
        Args:
            length: Length of random string to generate
            
        Returns:
            Random string containing letters and numbers
        """
        alphabet = string.ascii_lowercase + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    @classmethod
    def hash_api_key(cls, api_key: str) -> str:
        """Hash an API key for secure storage.
        
        Args:
            api_key: Raw API key to hash
            
        Returns:
            Hashed API key suitable for database storage
        """
        # Use bcrypt for secure hashing
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(api_key.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @classmethod
    def verify_api_key(cls, api_key: str, hashed_key: str) -> bool:
        """Verify an API key against its hash.
        
        Args:
            api_key: Raw API key to verify
            hashed_key: Stored hash to verify against
            
        Returns:
            True if API key matches hash, False otherwise
        """
        try:
            return bcrypt.checkpw(
                api_key.encode('utf-8'), 
                hashed_key.encode('utf-8')
            )
        except Exception as e:
            logger.error(f"Error verifying API key: {e}")
            return False
    
    @classmethod
    def extract_project_name(cls, api_key: str) -> Optional[str]:
        """Extract project name from API key.
        
        Args:
            api_key: API key to extract project name from
            
        Returns:
            Project name if found, None if invalid format
        """
        if not api_key.startswith(cls.PREFIX):
            return None
        
        # Remove prefix
        without_prefix = api_key[len(cls.PREFIX):]
        
        # Split on last underscore to separate project from random part
        parts = without_prefix.rsplit('_', 1)
        if len(parts) != 2:
            return None
        
        project_name, random_part = parts
        
        # Validate project name format
        if not cls.VALID_PROJECT_NAME_PATTERN.match(project_name):
            return None
        
        # Validate random part length (should be our expected length)
        if len(random_part) != cls.RANDOM_LENGTH:
            return None
        
        return project_name
    
    @classmethod
    def get_key_prefix(cls, api_key: str) -> Optional[str]:
        """Get the prefix part of an API key (for indexing).
        
        Args:
            api_key: API key to extract prefix from
            
        Returns:
            Key prefix (e.g., "kk_mobile_app") or None if invalid
        """
        project_name = cls.extract_project_name(api_key)
        if not project_name:
            return None
        
        return f"{cls.PREFIX}{project_name}"
    
    @classmethod
    def validate_api_key_format(cls, api_key: str) -> bool:
        """Validate API key format without checking database.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if format is valid, False otherwise
        """
        if not isinstance(api_key, str):
            return False
        
        # Check prefix
        if not api_key.startswith(cls.PREFIX):
            return False
        
        # Extract and validate project name
        project_name = cls.extract_project_name(api_key)
        if not project_name:
            return False
        
        # Check total length (prefix + project + underscore + random)
        expected_length = len(cls.PREFIX) + len(project_name) + 1 + cls.RANDOM_LENGTH
        if len(api_key) != expected_length:
            return False
        
        return True
    
    @classmethod
    def mask_api_key(cls, api_key: str, visible_chars: int = 8) -> str:
        """Mask an API key for safe display.
        
        Args:
            api_key: API key to mask
            visible_chars: Number of characters to show at the end
            
        Returns:
            Masked API key (e.g., "kk_mobile_***abc123")
        """
        if not api_key or len(api_key) <= visible_chars:
            return "*" * 12  # Generic mask
        
        prefix = cls.get_key_prefix(api_key)
        if prefix:
            # Show prefix + *** + last few chars
            visible_end = api_key[-visible_chars:]
            return f"{prefix}_***{visible_end}"
        else:
            # Fallback for invalid format
            visible_end = api_key[-visible_chars:]
            return f"***{visible_end}"


# Convenience functions for common operations
def generate_project_api_key(project_name: str) -> Tuple[str, str]:
    """Generate API key and its hash for a project.
    
    Args:
        project_name: Name of the project
        
    Returns:
        Tuple of (raw_api_key, hashed_api_key)
    """
    raw_key = APIKeyGenerator.generate_api_key(project_name)
    hashed_key = APIKeyGenerator.hash_api_key(raw_key)
    return raw_key, hashed_key


def validate_and_extract_project(api_key: str) -> Optional[str]:
    """Validate API key format and extract project name.
    
    Args:
        api_key: API key to validate
        
    Returns:
        Project name if valid, None if invalid
    """
    if not APIKeyGenerator.validate_api_key_format(api_key):
        return None
    
    return APIKeyGenerator.extract_project_name(api_key)