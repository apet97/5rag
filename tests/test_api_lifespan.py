from fastapi.testclient import TestClient

import clockify_rag.api as api_module


def test_create_app_lifespan_runs(monkeypatch):
    """create_app should initialize without raising during FastAPI lifespan."""

    monkeypatch.setattr(api_module, "ensure_index_ready", lambda retries=2: None)

    app = api_module.create_app()

    # entering the TestClient context runs the app lifespan (startup/shutdown)
    with TestClient(app):
        pass
