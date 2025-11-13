"""Tests for the shared LLM client abstraction."""

from clockify_rag.api_client import (
    MockLLMClient,
    get_llm_client,
    set_llm_client,
    reset_llm_client,
)
from clockify_rag import config


def test_mock_llm_client_embeddings_are_deterministic():
    client = MockLLMClient(embed_dim=16)
    vec1 = client.create_embedding("deterministic-text")
    vec2 = client.create_embedding("deterministic-text")
    assert vec1 == vec2
    assert len(vec1) == 16


def test_mock_llm_client_chat_completion_echoes_prompt():
    client = MockLLMClient()
    client.register_chat_response("ping?", "pong!")
    response = client.chat_completion(
        messages=[
            {"role": "user", "content": "ping?"},
        ],
        model=config.RAG_CHAT_MODEL,
    )
    assert response["message"]["content"] == "pong!"


def test_set_llm_client_overrides_global_instance():
    custom = MockLLMClient()
    set_llm_client(custom)
    try:
        assert get_llm_client() is custom
    finally:
        reset_llm_client()
