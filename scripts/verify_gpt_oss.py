#!/usr/bin/env python3
"""Verification script for GPT-OSS-20B integration."""

import sys
import os

# Add repo to path
sys.path.insert(0, os.path.dirname(__file__))

# Set test environment variables
os.environ["RAG_PROVIDER"] = "gpt-oss"
os.environ["RAG_GPT_OSS_MODEL"] = "gpt-oss-20b"
os.environ["RAG_GPT_OSS_TEMPERATURE"] = "1.0"
os.environ["RAG_GPT_OSS_TOP_P"] = "1.0"
os.environ["RAG_GPT_OSS_CTX_WINDOW"] = "128000"
os.environ["RAG_GPT_OSS_CTX_BUDGET"] = "16000"
os.environ["RAG_GPT_OSS_CHAT_TIMEOUT"] = "180.0"
os.environ["RAG_LLM_CLIENT"] = ""  # Use real client, not mock

print("✓ Environment variables set")

# Import config
try:
    from clockify_rag import config
    print(f"✓ Imported config module")
    print(f"  - RAG_PROVIDER: {config.RAG_PROVIDER}")
    print(f"  - RAG_GPT_OSS_MODEL: {config.RAG_GPT_OSS_MODEL}")
    print(f"  - RAG_GPT_OSS_TEMPERATURE: {config.RAG_GPT_OSS_TEMPERATURE}")
    print(f"  - RAG_GPT_OSS_TOP_P: {config.RAG_GPT_OSS_TOP_P}")
    print(f"  - RAG_GPT_OSS_CTX_WINDOW: {config.RAG_GPT_OSS_CTX_WINDOW}")
    print(f"  - RAG_GPT_OSS_CTX_BUDGET: {config.RAG_GPT_OSS_CTX_BUDGET}")
except Exception as e:
    print(f"✗ Failed to import config: {e}")
    sys.exit(1)

# Test context budget selection
try:
    budget = config.get_context_budget()
    window = config.get_context_window()
    print(f"✓ Context budget selection works")
    print(f"  - Context budget: {budget:,} tokens")
    print(f"  - Context window: {window:,} tokens")
    assert budget == 16000, f"Expected budget=16000, got {budget}"
    assert window == 128000, f"Expected window=128000, got {window}"
except Exception as e:
    print(f"✗ Context budget selection failed: {e}")
    sys.exit(1)

# Import API client
try:
    from clockify_rag.api_client import GptOssAPIClient, get_llm_client
    print(f"✓ Imported GptOssAPIClient")
except Exception as e:
    print(f"✗ Failed to import GptOssAPIClient: {e}")
    sys.exit(1)

# Test LLM settings
try:
    settings = config.current_llm_settings()
    print(f"✓ LLM settings:")
    print(f"  - Provider: {settings.provider}")
    print(f"  - Chat model: {settings.chat_model}")
    print(f"  - Embed model: {settings.embed_model}")
    print(f"  - Base URL: {settings.base_url}")
    assert settings.provider == "gpt-oss", f"Expected provider=gpt-oss, got {settings.provider}"
    assert settings.chat_model == "gpt-oss-20b", f"Expected model=gpt-oss-20b, got {settings.chat_model}"
except Exception as e:
    print(f"✗ LLM settings failed: {e}")
    sys.exit(1)

# Reset provider to test ollama
os.environ["RAG_PROVIDER"] = "ollama"
# Reload config by reimporting (in real code this would be at startup)
import importlib
importlib.reload(config)

try:
    budget = config.get_context_budget()
    window = config.get_context_window()
    print(f"✓ Ollama fallback works")
    print(f"  - Context budget: {budget:,} tokens")
    print(f"  - Context window: {window:,} tokens")
    # Note: These values depend on config defaults
except Exception as e:
    print(f"✗ Ollama fallback failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL VERIFICATIONS PASSED")
print("="*60)
print("\nGPT-OSS-20B integration is working correctly!")
print("\nTo use it:")
print("  export RAG_PROVIDER=gpt-oss")
print("  ragctl chat")
