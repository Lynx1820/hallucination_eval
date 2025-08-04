"""
Authentication utilities for HuggingFace integration.

This module provides elegant authentication handling for HuggingFace models,
eliminating the need for hardcoded tokens and manual login processes.
"""

import os
from typing import Optional
from pathlib import Path


def setup_huggingface_auth(token: Optional[str] = None) -> bool:
    """
    Set up HuggingFace authentication using environment variables or .env file.
    
    Args:
        token: Optional token to use. If None, will try environment variables.
        
    Returns:
        True if authentication was successful, False otherwise.
    """
    # Try to load from .env file if it exists
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        _load_env_file(env_file)
    
    # Get token from parameter, environment, or .env file
    if token is None:
        token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    
    if not token:
        print("Warning: No HuggingFace token found. Please set HUGGINGFACE_TOKEN environment variable.")
        return False
    
    # Set up HuggingFace environment
    os.environ['HF_TOKEN'] = token
    if hf_home := os.getenv('HF_HOME'):
        os.environ['HF_HOME'] = hf_home
    if hf_cache := os.getenv('HF_HUB_CACHE'):
        os.environ['HF_HUB_CACHE'] = hf_cache
    
    # Perform login
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("✓ HuggingFace authentication successful")
        return True
    except Exception as e:
        print(f"✗ HuggingFace authentication failed: {e}")
        return False


def _load_env_file(env_file: Path) -> None:
    """Load environment variables from .env file."""
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")


def get_cache_dir() -> str:
    """Get the configured HuggingFace cache directory."""
    return os.getenv('HF_HOME', os.getenv('HF_HUB_CACHE', '/tmp/huggingface'))


def is_authenticated() -> bool:
    """Check if HuggingFace authentication is available."""
    return bool(os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN'))