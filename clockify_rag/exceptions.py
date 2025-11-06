"""Custom exceptions for Clockify RAG system."""


class EmbeddingError(Exception):
    """Embedding generation failed."""
    pass


class LLMError(Exception):
    """LLM call failed."""
    pass


class IndexError(Exception):
    """Index loading or validation failed."""
    pass


class BuildError(Exception):
    """Knowledge base build failed."""
    pass
