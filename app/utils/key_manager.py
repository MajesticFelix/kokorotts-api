"""Key management utilities for CRUD operations on API keys and projects."""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from ..database.connection import get_session
from ..database.models import APIKey, Project, RateLimitTier
from ..utils.api_key_generator import generate_project_api_key, APIKeyGenerator
from ..models.api_requests import (
    CreateProjectRequest,
    CreateAPIKeyRequest,
    UpdateAPIKeyRequest,
    ProjectResponse,
    APIKeyResponse,
    APIKeyWithSecretResponse,
    ProjectUsageStats,
    APIKeyUsageStats
)

logger = logging.getLogger(__name__)


class KeyManager:
    """Manager class for API key and project operations."""
    
    @staticmethod
    def create_project(request: CreateProjectRequest) -> ProjectResponse:
        """Create a new project.
        
        Args:
            request: Project creation request
            
        Returns:
            Created project information
            
        Raises:
            ValueError: If project name already exists
            RuntimeError: If database operation fails
        """
        try:
            with next(get_session()) as session:
                if session is None:
                    raise RuntimeError("Database not available")
                
                # Check if project name already exists
                existing = session.query(Project).filter(
                    Project.name == request.name
                ).first()
                
                if existing:
                    raise ValueError(f"Project name '{request.name}' already exists")
                
                # Create new project
                project = Project(
                    name=request.name,
                    description=request.description
                )
                
                session.add(project)
                session.commit()
                session.refresh(project)
                
                logger.info(f"Created project: {project.name} (ID: {project.id})")
                
                return ProjectResponse(
                    id=project.id,
                    name=project.name,
                    description=project.description,
                    created_at=project.created_at,
                    total_requests=0,
                    total_characters=0,
                    api_key_count=0
                )
                
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error creating project: {e}")
            raise RuntimeError(f"Failed to create project: {str(e)}")
    
    @staticmethod
    def create_api_key(request: CreateAPIKeyRequest) -> APIKeyWithSecretResponse:
        """Create a new API key for a project.
        
        Args:
            request: API key creation request
            
        Returns:
            Created API key information including the secret
            
        Raises:
            ValueError: If project doesn't exist or validation fails
            RuntimeError: If database operation fails
        """
        try:
            with next(get_session()) as session:
                if session is None:
                    raise RuntimeError("Database not available")
                
                # Find or create project
                project = session.query(Project).filter(
                    Project.name == request.project_name
                ).first()
                
                if not project:
                    # Auto-create project if it doesn't exist
                    project = Project(
                        name=request.project_name,
                        description=f"Auto-created for API key"
                    )
                    session.add(project)
                    session.flush()  # Get the ID without committing
                
                # Generate API key
                raw_key, hashed_key = generate_project_api_key(request.project_name)
                key_prefix = APIKeyGenerator.get_key_prefix(raw_key)
                
                # Calculate expiration date
                expires_at = None
                if request.expires_in_days:
                    expires_at = datetime.utcnow() + timedelta(days=request.expires_in_days)
                
                # Create API key record
                api_key = APIKey(
                    key_hash=hashed_key,
                    key_prefix=key_prefix,
                    project_id=project.id,
                    description=request.description,
                    rate_limit_tier=request.rate_limit_tier,
                    expires_at=expires_at
                )
                
                session.add(api_key)
                session.commit()
                session.refresh(api_key)
                
                logger.info(f"Created API key: {key_prefix} for project {project.name}")
                
                return APIKeyWithSecretResponse(
                    id=api_key.id,
                    api_key=raw_key,  # Only shown once!
                    key_prefix=api_key.key_prefix,
                    project_id=api_key.project_id,
                    project_name=project.name,
                    description=api_key.description,
                    rate_limit_tier=api_key.rate_limit_tier,
                    is_active=api_key.is_active,
                    created_at=api_key.created_at,
                    last_used_at=api_key.last_used_at,
                    expires_at=api_key.expires_at,
                    usage_count=api_key.usage_count,
                    total_characters=api_key.total_characters
                )
                
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error creating API key: {e}")
            raise RuntimeError(f"Failed to create API key: {str(e)}")
    
    @staticmethod
    def list_projects(page: int = 1, page_size: int = 50) -> Tuple[List[ProjectResponse], int]:
        """List all projects with pagination.
        
        Args:
            page: Page number (1-based)
            page_size: Number of items per page
            
        Returns:
            Tuple of (projects list, total count)
        """
        try:
            with next(get_session()) as session:
                if session is None:
                    return [], 0
                
                # Get total count
                total_count = session.query(Project).count()
                
                # Get paginated results
                offset = (page - 1) * page_size
                projects = session.query(Project).order_by(
                    desc(Project.created_at)
                ).offset(offset).limit(page_size).all()
                
                # Build responses with API key counts
                project_responses = []
                for project in projects:
                    api_key_count = session.query(APIKey).filter(
                        APIKey.project_id == project.id
                    ).count()
                    
                    project_responses.append(ProjectResponse(
                        id=project.id,
                        name=project.name,
                        description=project.description,
                        created_at=project.created_at,
                        total_requests=project.total_requests,
                        total_characters=project.total_characters,
                        api_key_count=api_key_count
                    ))
                
                return project_responses, total_count
                
        except Exception as e:
            logger.error(f"Error listing projects: {e}")
            return [], 0
    
    @staticmethod
    def list_api_keys(
        project_name: Optional[str] = None,
        page: int = 1,
        page_size: int = 50
    ) -> Tuple[List[APIKeyResponse], int]:
        """List API keys with optional project filtering.
        
        Args:
            project_name: Optional project name filter
            page: Page number (1-based)
            page_size: Number of items per page
            
        Returns:
            Tuple of (API keys list, total count)
        """
        try:
            with next(get_session()) as session:
                if session is None:
                    return [], 0
                
                # Build query
                query = session.query(APIKey).join(Project)
                
                if project_name:
                    query = query.filter(Project.name == project_name)
                
                # Get total count
                total_count = query.count()
                
                # Get paginated results
                offset = (page - 1) * page_size
                api_keys = query.order_by(
                    desc(APIKey.created_at)
                ).offset(offset).limit(page_size).all()
                
                # Build responses
                responses = []
                for api_key in api_keys:
                    responses.append(APIKeyResponse(
                        id=api_key.id,
                        key_prefix=api_key.key_prefix,
                        project_id=api_key.project_id,
                        project_name=api_key.project.name,
                        description=api_key.description,
                        rate_limit_tier=api_key.rate_limit_tier,
                        is_active=api_key.is_active,
                        created_at=api_key.created_at,
                        last_used_at=api_key.last_used_at,
                        expires_at=api_key.expires_at,
                        usage_count=api_key.usage_count,
                        total_characters=api_key.total_characters
                    ))
                
                return responses, total_count
                
        except Exception as e:
            logger.error(f"Error listing API keys: {e}")
            return [], 0
    
    @staticmethod
    def get_api_key(key_id: UUID) -> Optional[APIKeyResponse]:
        """Get specific API key by ID.
        
        Args:
            key_id: API key ID
            
        Returns:
            API key information if found, None otherwise
        """
        try:
            with next(get_session()) as session:
                if session is None:
                    return None
                
                api_key = session.query(APIKey).join(Project).filter(
                    APIKey.id == key_id
                ).first()
                
                if not api_key:
                    return None
                
                return APIKeyResponse(
                    id=api_key.id,
                    key_prefix=api_key.key_prefix,
                    project_id=api_key.project_id,
                    project_name=api_key.project.name,
                    description=api_key.description,
                    rate_limit_tier=api_key.rate_limit_tier,
                    is_active=api_key.is_active,
                    created_at=api_key.created_at,
                    last_used_at=api_key.last_used_at,
                    expires_at=api_key.expires_at,
                    usage_count=api_key.usage_count,
                    total_characters=api_key.total_characters
                )
                
        except Exception as e:
            logger.error(f"Error getting API key: {e}")
            return None
    
    @staticmethod
    def update_api_key(key_id: UUID, request: UpdateAPIKeyRequest) -> Optional[APIKeyResponse]:
        """Update an API key.
        
        Args:
            key_id: API key ID
            request: Update request
            
        Returns:
            Updated API key information if successful, None otherwise
        """
        try:
            with next(get_session()) as session:
                if session is None:
                    return None
                
                api_key = session.query(APIKey).filter(
                    APIKey.id == key_id
                ).first()
                
                if not api_key:
                    return None
                
                # Update fields
                if request.description is not None:
                    api_key.description = request.description
                
                if request.rate_limit_tier is not None:
                    api_key.rate_limit_tier = request.rate_limit_tier
                
                if request.is_active is not None:
                    api_key.is_active = request.is_active
                
                session.commit()
                session.refresh(api_key)
                
                logger.info(f"Updated API key: {api_key.key_prefix}")
                
                return KeyManager.get_api_key(key_id)
                
        except Exception as e:
            logger.error(f"Error updating API key: {e}")
            return None
    
    @staticmethod
    def regenerate_api_key(key_id: UUID) -> Optional[APIKeyWithSecretResponse]:
        """Regenerate an API key (new secret, same settings).
        
        Args:
            key_id: API key ID
            
        Returns:
            New API key information if successful, None otherwise
        """
        try:
            with next(get_session()) as session:
                if session is None:
                    return None
                
                api_key = session.query(APIKey).join(Project).filter(
                    APIKey.id == key_id
                ).first()
                
                if not api_key:
                    return None
                
                # Generate new API key
                raw_key, hashed_key = generate_project_api_key(api_key.project.name)
                new_key_prefix = APIKeyGenerator.get_key_prefix(raw_key)
                
                # Update the key
                old_prefix = api_key.key_prefix
                api_key.key_hash = hashed_key
                api_key.key_prefix = new_key_prefix
                
                session.commit()
                session.refresh(api_key)
                
                logger.info(f"Regenerated API key: {old_prefix} -> {new_key_prefix}")
                
                return APIKeyWithSecretResponse(
                    id=api_key.id,
                    api_key=raw_key,  # Only shown once!
                    key_prefix=api_key.key_prefix,
                    project_id=api_key.project_id,
                    project_name=api_key.project.name,
                    description=api_key.description,
                    rate_limit_tier=api_key.rate_limit_tier,
                    is_active=api_key.is_active,
                    created_at=api_key.created_at,
                    last_used_at=api_key.last_used_at,
                    expires_at=api_key.expires_at,
                    usage_count=api_key.usage_count,
                    total_characters=api_key.total_characters
                )
                
        except Exception as e:
            logger.error(f"Error regenerating API key: {e}")
            return None
    
    @staticmethod
    def delete_api_key(key_id: UUID) -> bool:
        """Delete an API key.
        
        Args:
            key_id: API key ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with next(get_session()) as session:
                if session is None:
                    return False
                
                api_key = session.query(APIKey).filter(
                    APIKey.id == key_id
                ).first()
                
                if not api_key:
                    return False
                
                key_prefix = api_key.key_prefix
                session.delete(api_key)
                session.commit()
                
                logger.info(f"Deleted API key: {key_prefix}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting API key: {e}")
            return False
    
    @staticmethod
    def get_project_usage_stats(project_name: str) -> Optional[ProjectUsageStats]:
        """Get usage statistics for a project.
        
        Args:
            project_name: Project name
            
        Returns:
            Project usage statistics if found, None otherwise
        """
        try:
            with next(get_session()) as session:
                if session is None:
                    return None
                
                project = session.query(Project).filter(
                    Project.name == project_name
                ).first()
                
                if not project:
                    return None
                
                # Get API key counts
                total_keys = session.query(APIKey).filter(
                    APIKey.project_id == project.id
                ).count()
                
                active_keys = session.query(APIKey).filter(
                    APIKey.project_id == project.id,
                    APIKey.is_active == True
                ).count()
                
                # Get last activity
                last_activity = session.query(func.max(APIKey.last_used_at)).filter(
                    APIKey.project_id == project.id
                ).scalar()
                
                # Calculate days since creation
                days_since_creation = max(1, (datetime.utcnow() - project.created_at).days)
                
                return ProjectUsageStats(
                    project_id=project.id,
                    project_name=project.name,
                    total_requests=project.total_requests,
                    total_characters=project.total_characters,
                    api_key_count=total_keys,
                    active_api_key_count=active_keys,
                    days_since_creation=days_since_creation,
                    average_requests_per_day=project.total_requests / days_since_creation,
                    last_activity=last_activity
                )
                
        except Exception as e:
            logger.error(f"Error getting project usage stats: {e}")
            return None
    
    @staticmethod
    def get_available_rate_limit_tiers() -> Dict[str, Any]:
        """Get available rate limit tiers.
        
        Returns:
            Dictionary of available tiers with their limits
        """
        return RateLimitTier.TIERS