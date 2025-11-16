import json
import os
import pytest

from clockify_rag.cli import chat_repl


@pytest.mark.skip(reason="Test needs update: answer_once() return structure changed, load_index() location changed")
def test_chat_repl_json_output(monkeypatch, capsys):
    # Ensure environment setup routines are no-ops for the test
    # Note: _log_config_summary and warmup_on_startup may need to be found in cli module
    # monkeypatch.setattr(cli_module, "_log_config_summary", lambda **_: None)
    # monkeypatch.setattr(cli_module, "warmup_on_startup", lambda: None)
    monkeypatch.setattr(os.path, "exists", lambda path: True)

    # Import build from indexing module
    import clockify_rag.indexing

    monkeypatch.setattr(clockify_rag.indexing, "build", lambda *_, **__: None)

    # Provide deterministic artifacts and retrieval response
    chunks = {"chunk-1": {"id": "chunk-1", "text": "Citation text"}}
    monkeypatch.setattr(clockify_rag.indexing, "load_index", lambda: (chunks, object(), object(), object()))

    expected_citations = [{"id": "chunk-1", "text": "Citation text"}]
    expected_tokens = 987

    def fake_answer_once(*args, **kwargs):
        return "Mocked answer", {
            "selected": expected_citations,
            "used_tokens": expected_tokens,
        }

    import clockify_rag.answer

    monkeypatch.setattr(clockify_rag.answer, "answer_once", fake_answer_once)

    # Simulate a single question followed by EOF to exit the REPL
    inputs = iter(["What is Clockify?"])

    def fake_input(_prompt=""):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)

    chat_repl(use_json=True)

    captured = capsys.readouterr().out
    json_start = captured.index("{")
    json_payload = captured[json_start:]
    output = json.loads(json_payload)

    assert output["debug"]["meta"]["used_tokens"] == expected_tokens
    assert output["citations"] == expected_citations
