"""Tests for automatic model fallback functionality."""

import pytest
import requests
from unittest.mock import Mock, patch

from clockify_rag.api_client import (
    GptOssAPIClient,
    chat_completion,
    get_fallback_client,
    validate_models,
    reset_llm_client,
)
import clockify_rag.api_client as api_client_module
from clockify_rag.exceptions import LLMUnavailableError


@pytest.fixture(autouse=True)
def reset_client():
    """Reset the global LLM client before and after each test."""
    reset_llm_client()
    yield
    reset_llm_client()


@pytest.fixture
def mock_session():
    """Create a mock session for HTTP requests."""
    session = Mock()
    session.trust_env = False
    return session


@pytest.fixture
def enable_fallback(monkeypatch):
    """Enable fallback for testing."""
    # Patch both config module and api_client imports
    monkeypatch.setattr(api_client_module, "RAG_FALLBACK_ENABLED", True)
    monkeypatch.setattr(api_client_module, "RAG_FALLBACK_PROVIDER", "gpt-oss")
    monkeypatch.setattr(api_client_module, "RAG_FALLBACK_MODEL", "gpt-oss:20b")
    monkeypatch.setattr(api_client_module, "RAG_PROVIDER", "ollama")


@pytest.fixture
def disable_fallback(monkeypatch):
    """Disable fallback for testing."""
    # Reset clients before changing config to ensure cache is cleared
    reset_llm_client()
    # Patch api_client module import
    monkeypatch.setattr(api_client_module, "RAG_FALLBACK_ENABLED", False)


def test_fallback_client_creation_when_enabled(enable_fallback):
    """Test that fallback client is created when enabled."""
    fallback = get_fallback_client()
    assert fallback is not None
    assert isinstance(fallback, GptOssAPIClient)


def test_fallback_client_none_when_disabled(disable_fallback):
    """Test that fallback client is None when disabled."""
    # Reset again after config change to ensure cache is cleared
    reset_llm_client()
    fallback = get_fallback_client()
    assert fallback is None


def test_fallback_client_none_when_same_as_primary(monkeypatch):
    """Test that fallback client is None when same as primary provider."""
    # Reset clients before changing config to ensure cache is cleared
    reset_llm_client()
    # Patch api_client module imports
    monkeypatch.setattr(api_client_module, "RAG_FALLBACK_ENABLED", True)
    monkeypatch.setattr(api_client_module, "RAG_PROVIDER", "gpt-oss")
    monkeypatch.setattr(api_client_module, "RAG_FALLBACK_PROVIDER", "gpt-oss")

    # Reset again after config change to pick up new config
    reset_llm_client()
    fallback = get_fallback_client()
    assert fallback is None


@patch("clockify_rag.api_client.get_session")
def test_chat_completion_fallback_on_connection_error(mock_get_session, enable_fallback, monkeypatch):
    """Test that chat_completion falls back when primary model is unavailable."""
    # Mock primary client to raise connection error
    primary_session = Mock()
    primary_session.trust_env = False
    primary_session.post.side_effect = requests.exceptions.ConnectionError("Connection refused")

    # Mock fallback client to succeed
    fallback_session = Mock()
    fallback_session.trust_env = False
    fallback_response = Mock()
    fallback_response.json.return_value = {
        "message": {"role": "assistant", "content": "Fallback answer"},
        "model": "gpt-oss:20b",
    }
    fallback_response.raise_for_status = Mock()
    fallback_session.post.return_value = fallback_response

    # Return primary session first, then fallback session
    mock_get_session.side_effect = [primary_session, fallback_session]

    messages = [{"role": "user", "content": "test question"}]

    # Should succeed with fallback
    response = chat_completion(messages=messages)
    assert response["message"]["content"] == "Fallback answer"
    assert response["model"] == "gpt-oss:20b"


@patch("clockify_rag.api_client.get_session")
def test_chat_completion_fallback_on_timeout(mock_get_session, enable_fallback):
    """Test that chat_completion falls back when primary model times out."""
    # Mock primary client to raise timeout error
    primary_session = Mock()
    primary_session.trust_env = False
    primary_session.post.side_effect = requests.exceptions.Timeout("Read timeout")

    # Mock fallback client to succeed
    fallback_session = Mock()
    fallback_session.trust_env = False
    fallback_response = Mock()
    fallback_response.json.return_value = {
        "message": {"role": "assistant", "content": "Fallback after timeout"},
        "model": "gpt-oss:20b",
    }
    fallback_response.raise_for_status = Mock()
    fallback_session.post.return_value = fallback_response

    mock_get_session.side_effect = [primary_session, fallback_session]

    messages = [{"role": "user", "content": "test question"}]

    response = chat_completion(messages=messages)
    assert response["message"]["content"] == "Fallback after timeout"


@patch("clockify_rag.api_client.get_session")
def test_chat_completion_fallback_on_5xx_error(mock_get_session, enable_fallback):
    """Test that chat_completion falls back when primary model returns 5xx server error."""
    # Mock primary client to raise HTTPError with 503 status
    primary_session = Mock()
    primary_session.trust_env = False
    primary_response = Mock()
    primary_response.status_code = 503
    http_error = requests.exceptions.HTTPError("503 Service Unavailable")
    http_error.response = primary_response
    primary_session.post.side_effect = http_error

    # Mock fallback client to succeed
    fallback_session = Mock()
    fallback_session.trust_env = False
    fallback_response = Mock()
    fallback_response.json.return_value = {
        "message": {"role": "assistant", "content": "Fallback after 5xx"},
        "model": "gpt-oss:20b",
    }
    fallback_response.raise_for_status = Mock()
    fallback_session.post.return_value = fallback_response

    mock_get_session.side_effect = [primary_session, fallback_session]

    messages = [{"role": "user", "content": "test question"}]

    # Should succeed with fallback
    response = chat_completion(messages=messages)
    assert response["message"]["content"] == "Fallback after 5xx"
    assert response["model"] == "gpt-oss:20b"


@patch("clockify_rag.api_client.get_session")
def test_chat_completion_no_fallback_on_4xx_error(mock_get_session, enable_fallback):
    """Test that chat_completion does NOT fall back on 4xx client errors."""
    # Mock primary client to raise HTTPError with 404 status
    primary_session = Mock()
    primary_session.trust_env = False
    primary_response = Mock()
    primary_response.status_code = 404
    http_error = requests.exceptions.HTTPError("404 Not Found")
    http_error.response = primary_response
    primary_session.post.side_effect = http_error

    mock_get_session.return_value = primary_session

    messages = [{"role": "user", "content": "test question"}]

    # Should raise LLMError (not LLMUnavailableError) since 4xx is a client error
    from clockify_rag.exceptions import LLMError

    with pytest.raises(LLMError) as exc_info:
        chat_completion(messages=messages)

    assert "404" in str(exc_info.value)


@patch("clockify_rag.api_client.get_session")
def test_chat_completion_no_fallback_when_disabled(mock_get_session, disable_fallback):
    """Test that chat_completion raises error when fallback is disabled."""
    primary_session = Mock()
    primary_session.trust_env = False
    primary_session.post.side_effect = requests.exceptions.ConnectionError("Connection refused")

    mock_get_session.return_value = primary_session

    messages = [{"role": "user", "content": "test question"}]

    # Should raise LLMUnavailableError since fallback is disabled
    with pytest.raises(LLMUnavailableError):
        chat_completion(messages=messages)


@patch("clockify_rag.api_client.get_session")
def test_chat_completion_both_unavailable(mock_get_session, enable_fallback):
    """Test that chat_completion raises error when both primary and fallback fail."""
    # Both sessions fail
    failing_session = Mock()
    failing_session.trust_env = False
    failing_session.post.side_effect = requests.exceptions.ConnectionError("Connection refused")

    mock_get_session.return_value = failing_session

    messages = [{"role": "user", "content": "test question"}]

    # Should raise LLMUnavailableError with info about both failures
    with pytest.raises(LLMUnavailableError) as exc_info:
        chat_completion(messages=messages)

    error_message = str(exc_info.value)
    assert "primary and fallback" in error_message.lower()


@patch("clockify_rag.api_client.get_session")
def test_validate_models_server_reachable(mock_get_session):
    """Test validate_models when server is reachable."""
    session = Mock()
    session.trust_env = False

    # Mock /api/tags response
    response = Mock()
    response.json.return_value = {
        "models": [
            {"name": "qwen2.5:32b"},
            {"name": "nomic-embed-text:latest"},
            {"name": "gpt-oss:20b"},
            {"name": "llama3.2:3b"},
        ]
    }
    response.raise_for_status = Mock()
    session.get.return_value = response

    mock_get_session.return_value = session

    result = validate_models(log_warnings=False)

    assert result["server_reachable"] is True
    assert len(result["models_available"]) == 4
    assert "qwen2.5:32b" in result["models_available"]
    assert "gpt-oss:20b" in result["models_available"]
    assert result["chat_model"]["available"] is True
    assert result["embed_model"]["available"] is True


@patch("clockify_rag.api_client.get_session")
def test_validate_models_missing_model(mock_get_session):
    """Test validate_models when a required model is missing."""
    session = Mock()
    session.trust_env = False

    # Mock /api/tags response with missing chat model
    response = Mock()
    response.json.return_value = {
        "models": [
            {"name": "nomic-embed-text:latest"},
            {"name": "gpt-oss:20b"},
        ]
    }
    response.raise_for_status = Mock()
    session.get.return_value = response

    mock_get_session.return_value = session

    result = validate_models(log_warnings=False)

    assert result["server_reachable"] is True
    assert result["chat_model"]["available"] is False  # qwen2.5:32b missing
    assert result["embed_model"]["available"] is True
    assert result["all_required_available"] is False


@patch("clockify_rag.api_client.get_session")
def test_validate_models_server_unreachable(mock_get_session):
    """Test validate_models when server is unreachable."""
    session = Mock()
    session.trust_env = False
    session.get.side_effect = requests.exceptions.ConnectionError("Connection refused")

    mock_get_session.return_value = session

    result = validate_models(log_warnings=False)

    assert result["server_reachable"] is False
    assert result["all_required_available"] is False


def test_validate_config():
    """Test config.validate_config() basic functionality."""
    from clockify_rag.config import validate_config

    result = validate_config()

    assert "valid" in result
    assert "warnings" in result
    assert "errors" in result
    assert "config_snapshot" in result
    assert isinstance(result["warnings"], list)
    assert isinstance(result["errors"], list)

    # Check config snapshot contains expected keys
    snapshot = result["config_snapshot"]
    assert "llm_endpoint" in snapshot
    assert "provider" in snapshot
    assert "chat_model" in snapshot
    assert "fallback_enabled" in snapshot
