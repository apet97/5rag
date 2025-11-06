import json

import clockify_support_cli_final as cli


def test_log_query_records_non_zero_scores(tmp_path, monkeypatch):
    """Ensure log entries include non-zero dense/BM25 scores."""
    log_path = tmp_path / "rag_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    monkeypatch.setattr(cli, "QUERY_LOG_FILE", str(log_path))

    retrieved_chunks = [
        {"id": "chunk-1", "dense": 0.42, "bm25": 0.15, "hybrid": 0.30},
        {"id": "chunk-2", "dense": 0.10, "bm25": 0.05, "hybrid": 0.08},
    ]

    cli.log_query(
        query="What is Clockify?",
        answer="Clockify is a time tracker.",
        retrieved_chunks=retrieved_chunks,
        latency_ms=120,
        refused=False,
        metadata={"test": True},
    )

    log_lines = log_path.read_text().strip().splitlines()
    assert log_lines, "Log file should contain at least one entry"

    entry = json.loads(log_lines[-1])
    assert entry["retrieved_chunks"][0]["id"] == "chunk-1"
    assert entry["retrieved_chunks"][0]["dense"] > 0
    assert entry["retrieved_chunks"][0]["bm25"] > 0
    assert entry["chunk_scores"]["dense"][0] > 0
    assert entry["chunk_scores"]["bm25"][0] > 0
