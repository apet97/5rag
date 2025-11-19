"""Configuration loader with YAML file support and environment variable overrides.

Precedence (highest to lowest):
1. Environment variables (RAG_* prefix)
2. Config file (if specified)
3. Default config (config/default.yaml)
4. Hardcoded defaults in config.py
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Try to import PyYAML, but gracefully degrade if not available
try:
    import yaml

    HAS_YAML = True
except ImportError:
    yaml = None  # type: ignore
    HAS_YAML = False

logger = logging.getLogger(__name__)

# Global config cache
_loaded_config: Optional[Dict[str, Any]] = None
_config_file_path: Optional[str] = None


def find_config_file() -> Optional[Path]:
    """Find config file in standard locations.

    Search order:
    1. RAG_CONFIG_FILE environment variable
    2. config/default.yaml (relative to project root)
    3. ~/.config/clockify-rag/config.yaml

    Returns:
        Path to config file if found, None otherwise
    """
    # 1. Environment variable
    env_path = os.environ.get("RAG_CONFIG_FILE")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        logger.warning(f"RAG_CONFIG_FILE points to non-existent file: {env_path}")

    # 2. Project default config
    project_root = Path(__file__).parent.parent
    default_config = project_root / "config" / "default.yaml"
    if default_config.exists():
        return default_config

    # 3. User home config
    user_config = Path.home() / ".config" / "clockify-rag" / "config.yaml"
    if user_config.exists():
        return user_config

    return None


def load_yaml_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, searches standard locations.

    Returns:
        Configuration dictionary
    """
    if not HAS_YAML:
        logger.warning("PyYAML not installed. Cannot load YAML config files. " "Install with: pip install PyYAML")
        return {}

    if config_path is None:
        config_path = find_config_file()

    if config_path is None:
        logger.info("No config file found, using hardcoded defaults")
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)  # type: ignore
        logger.info(f"Loaded config from {config_path}")
        return config or {}
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}


def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to config.

    Environment variables take precedence over config file values.

    Mapping examples:
        RAG_OLLAMA_URL → config["llm"]["endpoint"]
        RAG_CHAT_MODEL → config["llm"]["chat_model"]
        RAG_EMBED_MODEL → config["llm"]["embed_model"]
        RAG_FALLBACK_MODEL → config["llm"]["fallback"]["model"]
        RAG_FALLBACK_ENABLED → config["llm"]["fallback"]["enabled"]
        RAG_TOP_K → config["retrieval"]["top_k"]
        RAG_PACK_TOP → config["retrieval"]["pack_top"]
        RAG_THRESHOLD → config["retrieval"]["threshold"]
        RAG_HYBRID_ALPHA → config["retrieval"]["alpha"]
        EMB_BACKEND → config["embedding"]["backend"]

    Args:
        config: Base configuration dictionary

    Returns:
        Configuration with environment variable overrides applied
    """
    # LLM configuration
    if env_url := os.environ.get("RAG_OLLAMA_URL") or os.environ.get("OLLAMA_URL"):
        config.setdefault("llm", {})["endpoint"] = env_url

    if env_chat_model := os.environ.get("RAG_CHAT_MODEL") or os.environ.get("GEN_MODEL"):
        config.setdefault("llm", {})["chat_model"] = env_chat_model

    if env_embed_model := os.environ.get("RAG_EMBED_MODEL") or os.environ.get("EMB_MODEL"):
        config.setdefault("llm", {})["embed_model"] = env_embed_model

    if env_fallback_model := os.environ.get("RAG_FALLBACK_MODEL") or os.environ.get("FALLBACK_MODEL"):
        config.setdefault("llm", {}).setdefault("fallback", {})["model"] = env_fallback_model

    if "RAG_FALLBACK_ENABLED" in os.environ:
        enabled = os.environ["RAG_FALLBACK_ENABLED"].lower() in ("1", "true", "yes", "on")
        config.setdefault("llm", {}).setdefault("fallback", {})["enabled"] = enabled

    # Timeouts
    if env_chat_read_t := os.environ.get("RAG_CHAT_READ_TIMEOUT"):
        try:
            config.setdefault("llm", {}).setdefault("timeouts", {})["chat_read"] = float(env_chat_read_t)
        except ValueError:
            logger.warning(f"Invalid RAG_CHAT_READ_TIMEOUT value: {env_chat_read_t}")

    # Retrieval configuration
    if env_top_k := os.environ.get("RAG_TOP_K"):
        try:
            config.setdefault("retrieval", {})["top_k"] = int(env_top_k)
        except ValueError:
            logger.warning(f"Invalid RAG_TOP_K value: {env_top_k}")

    if env_pack_top := os.environ.get("RAG_PACK_TOP"):
        try:
            config.setdefault("retrieval", {})["pack_top"] = int(env_pack_top)
        except ValueError:
            logger.warning(f"Invalid RAG_PACK_TOP value: {env_pack_top}")

    if env_threshold := os.environ.get("RAG_THRESHOLD"):
        try:
            config.setdefault("retrieval", {})["threshold"] = float(env_threshold)
        except ValueError:
            logger.warning(f"Invalid RAG_THRESHOLD value: {env_threshold}")

    if env_alpha := os.environ.get("RAG_HYBRID_ALPHA"):
        try:
            config.setdefault("retrieval", {})["alpha"] = float(env_alpha)
        except ValueError:
            logger.warning(f"Invalid RAG_HYBRID_ALPHA value: {env_alpha}")

    if env_mmr_lambda := os.environ.get("RAG_MMR_LAMBDA"):
        try:
            config.setdefault("retrieval", {})["mmr_lambda"] = float(env_mmr_lambda)
        except ValueError:
            logger.warning(f"Invalid RAG_MMR_LAMBDA value: {env_mmr_lambda}")

    # Embedding configuration
    if env_backend := os.environ.get("EMB_BACKEND"):
        config.setdefault("embedding", {})["backend"] = env_backend

    # Index configuration
    if env_use_ann := os.environ.get("USE_ANN"):
        faiss_enabled = env_use_ann.lower() not in ("none", "false", "0", "off")
        config.setdefault("index", {}).setdefault("faiss", {})["enabled"] = faiss_enabled

    # API configuration
    if env_port := os.environ.get("RAG_API_PORT"):
        try:
            config.setdefault("api", {})["port"] = int(env_port)
        except ValueError:
            logger.warning(f"Invalid RAG_API_PORT value: {env_port}")

    # Logging configuration
    if env_log_level := os.environ.get("RAG_LOG_LEVEL") or os.environ.get("LOG_LEVEL"):
        config.setdefault("logging", {})["level"] = env_log_level.upper()

    return config


def get_config(reload: bool = False, config_file: Optional[str] = None) -> Dict[str, Any]:
    """Get effective configuration with all overrides applied.

    Args:
        reload: Force reload from file (ignore cache)
        config_file: Optional path to config file (overrides search)

    Returns:
        Complete configuration dictionary
    """
    global _loaded_config, _config_file_path

    # Return cached config if available and not reloading
    if not reload and _loaded_config is not None and config_file == _config_file_path:
        return _loaded_config

    # Load base config from file
    config_path = Path(config_file) if config_file else None
    base_config = load_yaml_config(config_path)

    # Apply environment variable overrides
    effective_config = apply_env_overrides(base_config)

    # Cache result
    _loaded_config = effective_config
    _config_file_path = config_file

    return effective_config


def get_nested(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely get nested dictionary value.

    Example:
        get_nested(config, "llm", "endpoint", default="http://localhost:11434")

    Args:
        config: Configuration dictionary
        *keys: Nested keys to traverse
        default: Default value if key path doesn't exist

    Returns:
        Value at key path or default
    """
    try:
        value = config
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def export_effective_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Export effective configuration for debugging.

    Returns configuration with all environment overrides applied.
    Useful for 'ragctl config show' command.

    Args:
        config_file: Optional path to config file

    Returns:
        Configuration dictionary with metadata
    """
    config = get_config(reload=True, config_file=config_file)

    # Add metadata
    config_path = find_config_file() if not config_file else Path(config_file)

    metadata = {
        "_metadata": {
            "config_file": str(config_path) if config_path else None,
            "env_overrides_applied": True,
            "precedence": [
                "1. Environment variables (RAG_* prefix)",
                "2. Config file (if specified)",
                "3. Default config (config/default.yaml)",
                "4. Hardcoded defaults in config.py",
            ],
        }
    }

    return {**metadata, **config}


# Convenience functions for accessing common config values


def get_llm_endpoint() -> str:
    """Get LLM endpoint URL."""
    config = get_config()
    return get_nested(config, "llm", "endpoint", default="http://10.127.0.192:11434")


def get_chat_model() -> str:
    """Get primary chat model name."""
    config = get_config()
    return get_nested(config, "llm", "chat_model", default="qwen2.5:32b")


def get_embed_model() -> str:
    """Get embedding model name."""
    config = get_config()
    return get_nested(config, "llm", "embed_model", default="nomic-embed-text:latest")


def get_fallback_model() -> str:
    """Get fallback chat model name."""
    config = get_config()
    return get_nested(config, "llm", "fallback", "model", default="gpt-oss:20b")


def is_fallback_enabled() -> bool:
    """Check if fallback is enabled."""
    config = get_config()
    return get_nested(config, "llm", "fallback", "enabled", default=True)


def get_retrieval_params() -> Dict[str, Any]:
    """Get retrieval parameters."""
    config = get_config()
    return get_nested(
        config,
        "retrieval",
        default={
            "top_k": 15,
            "pack_top": 8,
            "threshold": 0.25,
            "alpha": 0.5,
            "mmr_lambda": 0.75,
            "faiss_multiplier": 3,
        },
    )
