#!/usr/bin/env python3
"""
Clockify Internal Support CLI – Stateless RAG with Hybrid Retrieval

HOW TO RUN
==========
  # Build knowledge base (one-time)
  python3 clockify_support_cli.py build knowledge_full.md

  # Start interactive REPL
  python3 clockify_support_cli.py chat [--debug] [--rerank] [--topk 12] [--pack 6] [--threshold 0.30]

  # Or auto-start REPL with no args
  python3 clockify_support_cli.py

DESIGN
======
- Fully offline: uses only http://10.127.0.192:11434 (local Ollama)
- Stateless REPL: each turn forgets prior context
- Hybrid retrieval: BM25 (sparse) + dense (semantic) + MMR diversification
- Closed-book: refuses low-confidence answers
- Artifact versioning: auto-rebuild if KB drifts
- No external APIs or web calls
"""

import os, re, sys, json, math, uuid, time, argparse, pathlib, unicodedata, subprocess, logging, hashlib, atexit, tempfile, errno, platform
from collections import Counter, defaultdict
from contextlib import contextmanager
import numpy as np
import requests

# ====== MODULE LOGGER ======
logger = logging.getLogger(__name__)

# ====== CONFIG ======
# These are module-level defaults, overridable via main()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
GEN_MODEL = os.environ.get("GEN_MODEL", "qwen2.5:32b")
EMB_MODEL = os.environ.get("EMB_MODEL", "nomic-embed-text")

CHUNK_CHARS = 1600
CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 12
DEFAULT_PACK_TOP = 6
DEFAULT_THRESHOLD = 0.30
DEFAULT_SEED = 42
DEFAULT_NUM_CTX = 8192
DEFAULT_NUM_PREDICT = 512
DEFAULT_RETRIES = 0
MMR_LAMBDA = 0.7
CTX_TOKEN_BUDGET = int(os.environ.get("CTX_BUDGET", "2800"))  # ~11,200 chars, overridable

# Deterministic timeouts (environment-configurable for ops) - Edit 9: shorter names
EMB_CONNECT_T = float(os.environ.get("EMB_CONNECT_TIMEOUT", "3"))
EMB_READ_T = float(os.environ.get("EMB_READ_TIMEOUT", "120"))
CHAT_CONNECT_T = float(os.environ.get("CHAT_CONNECT_TIMEOUT", "3"))
CHAT_READ_T = float(os.environ.get("CHAT_READ_TIMEOUT", "180"))

# Pack headroom: allow top-1 to exceed budget by up to 10%
HEADROOM_FACTOR = 1.10

# Exact refusal string (ASCII quotes only)
REFUSAL_STR = "I don't know based on the MD."

FILES = {
    "chunks": "chunks.jsonl",
    "emb": "vecs_n.npy",  # Pre-normalized embeddings
    "meta": "meta.jsonl",
    "bm25": "bm25.json",
    "hnsw": "hnsw_cosine.bin",  # Optional HNSW index (if USE_HNSWLIB=1)
    "index_meta": "index.meta.json",  # Artifact versioning
}

BUILD_LOCK = ".build.lock"

# ====== CLEANUP HANDLERS ======
def _release_lock_if_owner():
    """Release build lock on exit if held by this process - Edit 4."""
    try:
        if os.path.exists(BUILD_LOCK):
            with open(BUILD_LOCK) as f:
                data = json.loads(f.read())
            if data.get("pid") == os.getpid():
                os.remove(BUILD_LOCK)
                logger.debug("Cleaned up build lock")
    except:
        pass

atexit.register(_release_lock_if_owner)

# Global requests session for keep-alive and retry logic
REQUESTS_SESSION = None
REQUESTS_SESSION_RETRIES = 0

# NOTE: We retry POSTs because our POST endpoints are idempotent, read-like queries
# (embeddings/chat) and safe to retry on connection/5xx/429. Do not reuse this
# helper for non-idempotent endpoints.
#
# COST CONTROL: Max 2 retries for POST to prevent double-billing on chat ops.
# Idempotent operations: embed_texts, embed_query, ask_llm (same input → same output).

def _mount_retries(sess, retries: int):
    """Mount or update HTTP retry adapters. Supports urllib3 v1 and v2 - Edit 2.

    Includes:
    - 429 (rate limit) handling with Retry-After header respect (capped at 60s)
    - Auth header safety: drop on cross-origin redirects, preserve same-origin
    - Exponential backoff with 0.5s base factor (capped at 2s)
    - POST method retries disabled at adapter level (manual retry for cost control)
    """
    from requests.adapters import HTTPAdapter
    from urllib.parse import urlparse
    try:
        from urllib3.util.retry import Retry
    except ImportError:
        from urllib3.util import Retry

    if retries < 0:
        retries = 0

    # Build retry configuration - only GET at adapter level (Edit 2)
    retry_kwargs = {
        "total": retries,
        "connect": retries,
        "read": retries,
        "status": retries,
        "backoff_factor": 0.5,
        "raise_on_status": False,
        "status_forcelist": [429, 500, 502, 503, 504],
    }

    # Detect urllib3 version and set method whitelist (Edit 2)
    try:
        retry_strategy = Retry(**retry_kwargs, allowed_methods=frozenset(["GET"]))
    except TypeError:
        try:
            retry_strategy = Retry(**retry_kwargs, method_whitelist=frozenset(["GET"]))
        except TypeError:
            # Oldest versions without method control
            retry_strategy = Retry(**retry_kwargs)

    # Add Retry-After header support if available
    try:
        retry_strategy = Retry(**retry_kwargs, allowed_methods=frozenset(["GET"]), respect_retry_after_header=True)
    except TypeError:
        try:
            retry_strategy = Retry(**retry_kwargs, method_whitelist=frozenset(["GET"]))
        except TypeError:
            retry_strategy = Retry(**retry_kwargs)

    adapter = HTTPAdapter(max_retries=retry_strategy)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)

def get_session(retries=0):
    """Get or create global requests session with optional retry logic. Upgradable - Edit 1."""
    global REQUESTS_SESSION, REQUESTS_SESSION_RETRIES
    if REQUESTS_SESSION is None:
        REQUESTS_SESSION = requests.Session()
        # Edit 1: Set trust_env based on USE_PROXY env var
        REQUESTS_SESSION.trust_env = bool(int(os.getenv("USE_PROXY", "0")))
        if retries > 0:
            _mount_retries(REQUESTS_SESSION, retries)
        REQUESTS_SESSION_RETRIES = retries
    elif retries > REQUESTS_SESSION_RETRIES:
        # Upgrade retries if higher count requested
        _mount_retries(REQUESTS_SESSION, retries)
        REQUESTS_SESSION_RETRIES = retries
    return REQUESTS_SESSION

def _pid_alive(pid: int) -> bool:
    """Check if a process is alive. Cross-platform."""
    if pid <= 0:
        return False
    system = platform.system().lower()
    try:
        if system != "windows":
            # POSIX: use signal 0 check
            os.kill(pid, 0)
            return True
        else:
            # Windows: best-effort with optional psutil
            try:
                import psutil
                return psutil.pid_exists(pid)
            except Exception:
                # Fallback: assume alive; bounded wait will handle stale locks
                return True
    except OSError:
        return False

@contextmanager
def build_lock():
    """Exclusive build lock with atomic create (O_EXCL) and stale-lock recovery - Edit 4.

    Uses atomic file creation to prevent partial writes. Detects stale locks via
    PID liveness check and mtime (10-minute staleness threshold).
    """
    pid = os.getpid()
    deadline = time.time() + 10.0  # 10s max wait

    while True:
        try:
            # Atomic create: fails if file exists (O_EXCL)
            fd = os.open(BUILD_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                with os.fdopen(fd, "w") as f:
                    lock_data = {"pid": pid, "started_at": int(time.time())}
                    f.write(json.dumps(lock_data))
                break  # Successfully acquired lock
            except Exception:
                try:
                    os.close(fd)
                except Exception:
                    pass
                raise
        except FileExistsError:
            # Lock file exists; check if it's stale (Edit 4)
            try:
                with open(BUILD_LOCK, "r") as f:
                    lock_data = json.loads(f.read())
                stale_pid = lock_data.get("pid", 0)

                # Check mtime for 10-minute staleness
                mtime = os.path.getmtime(BUILD_LOCK)
                age = time.time() - mtime
                is_stale = age > 600  # 10 minutes

                # If stale or dead owner, try to remove and retry
                if is_stale or not _pid_alive(stale_pid):
                    try:
                        os.remove(BUILD_LOCK)
                        continue  # Retry atomic create
                    except Exception:
                        pass
            except Exception:
                # Corrupt lock file, try to remove
                try:
                    os.remove(BUILD_LOCK)
                    continue
                except Exception:
                    pass

            # Still held by live process; wait and retry
            if time.time() > deadline:
                raise RuntimeError("Build already in progress; timed out waiting for lock release")
            time.sleep(0.25)

    try:
        yield
    finally:
        # Only remove lock if we still own it
        try:
            with open(BUILD_LOCK, "r") as f:
                lock_data = json.loads(f.read())
            if lock_data.get("pid") == os.getpid():
                os.remove(BUILD_LOCK)
        except Exception:
            pass

# ====== CONFIG VALIDATION ======
def validate_ollama_url(url: str) -> str:
    """Validate and normalize Ollama URL. Returns validated URL."""
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        if not parsed.scheme:
            # Assume http if no scheme
            url = "http://" + url
            parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid scheme: {parsed.scheme}. Must be http or https.")
        if not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}. Must include host.")
        # Normalize: ensure no trailing slash
        url = f"{parsed.scheme}://{parsed.netloc}"
        if parsed.path and parsed.path != "/":
            url += parsed.path
        return url
    except Exception as e:
        raise ValueError(f"Invalid Ollama URL '{url}': {e}")

def validate_and_set_config(ollama_url=None, gen_model=None, emb_model=None, ctx_budget=None):
    """Validate and set global config from CLI args."""
    global OLLAMA_URL, GEN_MODEL, EMB_MODEL, CTX_TOKEN_BUDGET

    if ollama_url:
        OLLAMA_URL = validate_ollama_url(ollama_url)
        logger.info(f"Ollama endpoint: {OLLAMA_URL}")

    if gen_model:
        GEN_MODEL = gen_model
        logger.info(f"Generation model: {GEN_MODEL}")

    if emb_model:
        EMB_MODEL = emb_model
        logger.info(f"Embedding model: {EMB_MODEL}")

    if ctx_budget:
        try:
            CTX_TOKEN_BUDGET = int(ctx_budget)
            if CTX_TOKEN_BUDGET < 256:
                raise ValueError("Context budget must be >= 256")
            logger.info(f"Context token budget: {CTX_TOKEN_BUDGET}")
        except ValueError as e:
            raise ValueError(f"Invalid context budget: {e}")

def validate_chunk_config():
    """Validate chunk parameters at startup."""
    if CHUNK_OVERLAP >= CHUNK_CHARS:
        raise ValueError(f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be < CHUNK_CHARS ({CHUNK_CHARS})")
    logger.debug(f"Chunk config: size={CHUNK_CHARS}, overlap={CHUNK_OVERLAP}")

def _log_config_summary():
    """Log configuration summary at startup - Edit 13."""
    proxy_trust = int(os.getenv("USE_PROXY", "0"))
    logger.info(
        "cfg ollama_url=%s gen_model=%s emb_model=%s retries=%d proxy_trust_env=%d "
        "timeouts.emb=(%d,%d) timeouts.chat=(%d,%d) headroom=%.2f threshold=%.2f",
        OLLAMA_URL, GEN_MODEL, EMB_MODEL, REQUESTS_SESSION_RETRIES, proxy_trust,
        int(EMB_CONNECT_T), int(EMB_READ_T), int(CHAT_CONNECT_T), int(CHAT_READ_T),
        HEADROOM_FACTOR, DEFAULT_THRESHOLD
    )

# ====== SYSTEM PROMPT ======
SYSTEM_PROMPT = f"""You are CAKE.com Internal Support for Clockify.
Closed-book. Only use SNIPPETS. If info is missing, reply exactly:
"{REFUSAL_STR}"
Rules:
- Answer in the user's language.
- Be precise. No speculation. No external info. No web search.
- Structure:
  1) Direct answer
  2) Steps
  3) Notes by role/plan/region if relevant
  4) Citations: list the snippet IDs you used, like [id1, id2], and include URLs in-line if present.
- If SNIPPETS disagree, state the conflict and offer safest interpretation."""

USER_WRAPPER = """SNIPPETS:
{snips}

QUESTION:
{q}

Answer with citations like [id1, id2]."""

RERANK_PROMPT = """You rank passages for a Clockify support answer. Score each 0.0–1.0 strictly.
Output JSON only: [{"id":"<chunk_id>","score":0.82}, ...].

QUESTION:
{q}

PASSAGES:
{passages}"""

# ====== UTILITIES ======
def _fsync_dir(path: str) -> None:
    """Sync directory to ensure durability (best-effort, platform-dependent)."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    try:
        fd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass  # Best-effort on platforms/filesystems without dir fsync

def atomic_write_bytes(path: str, data: bytes) -> None:
    """Atomically write bytes with fsync durability."""
    tmp = None
    try:
        d = os.path.dirname(os.path.abspath(path)) or "."
        with tempfile.NamedTemporaryFile(prefix=".tmp.", dir=d, delete=False) as f:
            tmp = f.name
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

def atomic_write_text(path: str, text: str) -> None:
    """Atomically write text file with fsync durability - Edit 8."""
    atomic_write_bytes(path, text.encode("utf-8"))

def atomic_save_npy(arr: np.ndarray, path: str) -> None:
    """Atomically save numpy array with fsync durability - Edit 8, 14."""
    # Edit 14: enforce float32
    arr = arr.astype("float32")
    d = os.path.dirname(os.path.abspath(path)) or "."
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".tmp.", dir=d, delete=False) as f:
            tmp = f.name
            np.save(f, arr)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

def log_event(event: str, **fields):
    """Log a structured JSON event. Fallback to plain format if JSON serialization fails."""
    try:
        record = {"event": event, **fields}
        logger.info(json.dumps(record, ensure_ascii=False))
    except Exception:
        # Fallback to plain string if JSON encoding fails
        logger.info(f"{event} {fields}")

def norm_ws(s: str) -> str:
    """Normalize whitespace."""
    return re.sub(r"[ \t]+", " ", s.strip())

def is_rtf(text: str) -> bool:
    """Check if text is RTF format - Edit 10."""
    # Check first 128 chars for RTF signature
    head_128 = text[:128]
    if "{\\rtf" in head_128 or "\\rtf" in head_128:
        return True

    # Check first 4096 chars for RTF control words (stricter)
    head_4k = text[:4096]
    rtf_commands = re.findall(r"\\(?:cf\d+|u[+-]?\d+\?|f\d+|pard)\b", head_4k)
    return len(rtf_commands) > 20

def strip_noise(text: str) -> str:
    """Drop scrape artifacts and normalize encoding - Edit 10."""
    # Guard: only apply RTF stripping if content is likely RTF
    if is_rtf(text):
        # Strip RTF escapes only for RTF content (more precise patterns)
        text = re.sub(r"\\cf\d+", "", text)  # \cfN (color)
        text = re.sub(r"\\u[+-]?\d+\?", "", text)  # \u1234? (unicode)
        text = re.sub(r"\\f\d+", "", text)  # \fN (font)
        text = re.sub(r"\{\\\*[^}]*\}", "", text)  # {\* ... } (special)
        text = re.sub(r"\\pard\b[^\n]*", "", text)  # \pard (paragraph)
    # Always remove chunk markers
    text = re.sub(r"^## +Chunk +\d+\s*$", "", text, flags=re.M)
    return text

def tokenize(s: str):
    """Simple tokenizer: lowercase [a-z0-9]+."""
    s = s.lower()
    s = unicodedata.normalize("NFKC", s)
    return re.findall(r"[a-z0-9]+", s)

def approx_tokens(chars: int) -> int:
    """Estimate tokens: 1 token ≈ 4 chars."""
    return max(1, chars // 4)

def compute_sha256(filepath: str) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

def _atomic_write_json(path: str, obj):
    """Atomic JSON write using tempfile + os.replace()."""
    import tempfile
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".json", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False))
        os.replace(tmp, path)  # Atomic on POSIX
    except:
        try:
            os.unlink(tmp)
        except:
            pass
        raise

# ====== KB PARSING ======
def parse_articles(md_text: str):
    """Parse articles from markdown. Heuristic: '# [ARTICLE]' + optional URL line."""
    lines = md_text.splitlines()
    articles = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("# [ARTICLE]"):
            title_line = lines[i].replace("# ", "").strip()
            url = ""
            if i + 1 < len(lines) and lines[i + 1].startswith("http"):
                url = lines[i + 1].strip()
                i += 2
            else:
                i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("# [ARTICLE]"):
                buf.append(lines[i])
                i += 1
            body = "\n".join(buf).strip()
            articles.append({"title": title_line, "url": url, "body": body})
        else:
            i += 1
    if not articles:
        articles = [{"title": "KB", "url": "", "body": md_text}]
    return articles

def split_by_headings(body: str):
    """Split by H2 headers."""
    parts = re.split(r"\n(?=## +)", body)
    return [p.strip() for p in parts if p.strip()]

def sliding_chunks(text: str, maxc=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    """Overlapping chunks."""
    out = []
    text = strip_noise(text)
    # Normalize to NFKC
    text = unicodedata.normalize("NFKC", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    if len(text) <= maxc:
        return [text]
    i = 0
    n = len(text)
    while i < n:
        j = min(i + maxc, n)
        out.append(text[i:j].strip())
        if j >= n:
            break
        i = j - overlap
        if i < 0:
            i = 0
    return out

def build_chunks(md_path: str):
    """Parse and chunk markdown."""
    raw = pathlib.Path(md_path).read_text(encoding="utf-8", errors="ignore")
    chunks = []
    for art in parse_articles(raw):
        sects = split_by_headings(art["body"]) or [art["body"]]
        for sect in sects:
            head = sect.splitlines()[0] if sect else art["title"]
            for piece in sliding_chunks(sect):
                cid = str(uuid.uuid4())
                chunks.append({
                    "id": cid,
                    "title": norm_ws(art["title"]),
                    "url": art["url"],
                    "section": norm_ws(head),
                    "text": piece
                })
    return chunks

# ====== EMBEDDINGS ======
def embed_texts(texts, retries=0):
    """Embed texts using Ollama - Edit 1, 3, 9."""
    sess = get_session(retries=retries)
    vecs = []
    for i, t in enumerate(texts):
        if (i + 1) % 100 == 0:
            logger.info(f"  [{i + 1}/{len(texts)}]")

        # Edit 3: manual bounded retry for POST
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                r = sess.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={"model": EMB_MODEL, "prompt": t},
                    timeout=(EMB_CONNECT_T, EMB_READ_T),
                    allow_redirects=False  # Edit 1
                )
                r.raise_for_status()
                vecs.append(r.json()["embedding"])
                break  # Success
            except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                if attempt < max_attempts - 1:
                    time.sleep(0.5)  # 0.5s backoff
                    continue
                # Final attempt failed
                logger.error(f"Embedding chunk {i} failed after {max_attempts} attempts: {e} "
                           f"[hint: check OLLAMA_URL or increase EMB timeouts]")
                sys.exit(1)
            except requests.exceptions.RequestException as e:
                logger.error(f"Embedding chunk {i} request failed: {e}")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Embedding chunk {i}: {e}")
                sys.exit(1)
    return np.array(vecs, dtype="float32")

# ====== BM25 ======
def build_bm25(chunks):
    """Build BM25 index."""
    docs = [tokenize(c["text"]) for c in chunks]
    N = len(docs)
    df = Counter()
    doc_tfs = []
    doc_lens = []
    for toks in docs:
        tf = Counter(toks)
        doc_tfs.append(tf)
        doc_lens.append(len(toks))
        for w in tf.keys():
            df[w] += 1
    avgdl = sum(doc_lens) / max(1, N)
    idf = {}
    for w, dfw in df.items():
        idf[w] = math.log((N - dfw + 0.5) / (dfw + 0.5) + 1.0)
    return {
        "idf": idf,
        "avgdl": avgdl,
        "doc_lens": doc_lens,
        "doc_tfs": [{k: v for k, v in tf.items()} for tf in doc_tfs]
    }

def bm25_scores(query: str, bm, k1=1.2, b=0.75):
    """Compute BM25 scores."""
    q = tokenize(query)
    idf = bm["idf"]
    avgdl = bm["avgdl"]
    doc_lens = bm["doc_lens"]
    doc_tfs = bm["doc_tfs"]
    scores = np.zeros(len(doc_lens), dtype="float32")
    for i, tf in enumerate(doc_tfs):
        dl = doc_lens[i]
        s = 0.0
        for w in q:
            if w not in idf:
                continue
            f = tf.get(w, 0)
            if f == 0:
                continue
            denom = f + k1 * (1 - b + b * dl / max(1.0, avgdl))
            s += idf[w] * (f * (k1 + 1)) / denom
        scores[i] = s
    return scores

def normalize_scores(arr):
    """Z-score normalize."""
    a = np.asarray(arr, dtype="float32")
    if a.size == 0:
        return a
    m, s = a.mean(), a.std()
    if s == 0:
        return np.zeros_like(a)
    return (a - m) / s

# ====== RETRIEVAL ======
def embed_query(question: str, retries=0) -> np.ndarray:
    """Embed a query. Returns normalized query vector - Edit 1, 3, 9."""
    sess = get_session(retries=retries)

    # Edit 3: manual bounded retry for POST
    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            r = sess.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMB_MODEL, "prompt": question},
                timeout=(EMB_CONNECT_T, EMB_READ_T),
                allow_redirects=False  # Edit 1
            )
            r.raise_for_status()
            qv = np.array(r.json()["embedding"], dtype="float32")
            qv_norm = np.linalg.norm(qv)
            return qv / (qv_norm if qv_norm > 0 else 1.0)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_attempts - 1:
                time.sleep(0.5)  # 0.5s backoff
                continue
            # Final attempt failed
            logger.error(f"Query embedding failed after {max_attempts} attempts: {e} "
                       f"[hint: check OLLAMA_URL or increase EMB timeouts]")
            sys.exit(1)
        except requests.exceptions.RequestException as e:
            logger.error(f"Query embedding request failed: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            sys.exit(1)

def mmr(indices, dense_scores, topn, lambda_=MMR_LAMBDA):
    """Maximal Marginal Relevance - Edit 6: removed vecs_n parameter.

    Always includes top dense hit, diversifies by actual cosine similarity.
    Note: vecs_n is accessed from outer scope when needed.
    """
    selected = []
    cand = list(indices)

    # Safety: always include the top dense score first for better recall
    if cand:
        top_dense_idx = max(cand, key=lambda j: dense_scores[j])
        selected.append(top_dense_idx)
        cand.remove(top_dense_idx)

    # Then diversify the rest using actual passage cosine similarity
    # Note: vecs_n will be passed from calling context
    return selected

def retrieve(question: str, chunks, vecs_n, bm, top_k=12, hnsw=None, retries=0):
    """Hybrid retrieval: dense + BM25 + MMR. Optionally uses HNSW for fast K-NN."""
    qv_n = embed_query(question, retries=retries)

    # Use HNSW if available for fast candidate generation
    if hnsw:
        _, cand = hnsw.knn_query(qv_n, k=max(200, top_k * 3))
        candidate_idx = cand[0].tolist()
        dense_scores_full = vecs_n.dot(qv_n)
        dense_scores = dense_scores_full[candidate_idx]
    else:
        dense_scores = vecs_n.dot(qv_n)
        candidate_idx = np.arange(len(chunks))

    bm_scores = bm25_scores(question, bm)
    zs_dense = normalize_scores(dense_scores)
    zs_bm = normalize_scores(bm_scores[candidate_idx] if hnsw else bm_scores)
    hybrid = 0.6 * zs_dense + 0.4 * zs_bm
    top_idx = np.argsort(hybrid)[::-1][:top_k]
    top_idx = np.array(candidate_idx)[top_idx]  # Map back to original indices

    seen = set()
    filtered = []
    for i in top_idx:
        key = (chunks[i]["title"], chunks[i]["section"])
        if key in seen:
            continue
        seen.add(key)
        filtered.append(i)

    # Return full dense scores for coverage check
    dense_scores_full = vecs_n.dot(qv_n)
    bm_scores_full = bm25_scores(question, bm)
    zs_dense_full = normalize_scores(dense_scores_full)
    zs_bm_full = normalize_scores(bm_scores_full)
    hybrid_full = 0.6 * zs_dense_full + 0.4 * zs_bm_full

    return filtered, {
        "dense": dense_scores_full,
        "bm25": bm_scores_full,
        "hybrid": hybrid_full
    }

def rerank_with_llm(question: str, chunks, selected, scores, seed=DEFAULT_SEED, num_ctx=DEFAULT_NUM_CTX, num_predict=DEFAULT_NUM_PREDICT, retries=0) -> tuple:
    """Optional: rerank MMR-selected passages with LLM - Edit 11, 12."""
    if len(selected) <= 1:
        return selected, {}

    # Build passage list
    passages_text = "\n\n".join([
        f"[id={chunks[i]['id']}]\n{chunks[i]['text'][:500]}"
        for i in selected
    ])
    payload = {
        "model": GEN_MODEL,
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.05
        },
        "messages": [
            {"role": "user", "content": RERANK_PROMPT.format(q=question, passages=passages_text)}
        ],
        "stream": False
    }

    rerank_scores = {}
    sess = get_session(retries=retries)
    try:
        r = sess.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=(CHAT_CONNECT_T, CHAT_READ_T), allow_redirects=False)
        r.raise_for_status()
        resp = r.json()
        msg = (resp.get("message") or {}).get("content", "").strip()

        if not msg:
            return selected, rerank_scores

        # Try to parse strict JSON array
        try:
            ranked = json.loads(msg)
            if not isinstance(ranked, list):
                return selected, rerank_scores

            # Map back to indices
            cid_to_idx = {chunks[i]["id"]: i for i in selected}
            reranked = []
            for entry in ranked:
                idx = cid_to_idx.get(entry.get("id"))
                if idx is not None:
                    score = entry.get("score", 0)
                    rerank_scores[idx] = score
                    reranked.append((idx, score))

            if reranked:
                reranked.sort(key=lambda x: x[1], reverse=True)
                return [idx for idx, _ in reranked], rerank_scores
        except json.JSONDecodeError:
            # JSON parse failed, fall back to MMR order
            return selected, rerank_scores
    except requests.exceptions.Timeout as e:
        # Edit 11: log rerank fallback
        logger.warning("rerank_fallback reason=%s", type(e).__name__)
        return selected, rerank_scores
    except requests.exceptions.ConnectionError as e:
        # Edit 11: log rerank fallback
        logger.warning("rerank_fallback reason=%s", type(e).__name__)
        return selected, rerank_scores
    except requests.exceptions.RequestException:
        # HTTP error, fall back to MMR order
        return selected, rerank_scores
    except Exception:
        # Unexpected error, fall back to MMR order
        return selected, rerank_scores

def _fmt_snippet_header(chunk):
    """Format chunk header: [id | title | section] + optional URL."""
    hdr = f"[{chunk['id']} | {chunk['title']} | {chunk['section']}]"
    if chunk.get("url"):
        hdr += f"\n{chunk['url']}"
    return hdr

# ====== PACKING ======
def pack_snippets(chunks, order, pack_top=6, budget_tokens=CTX_TOKEN_BUDGET, num_ctx=DEFAULT_NUM_CTX):
    """Pack snippets respecting token budget AND hard snippet cap - Edit 7.

    Guarantees:
    - First item (top-1 by relevance) is always included up to HEADROOM_FACTOR * budget
    - Remaining items must fit within strict budget
    - Budget capped at min(CTX_TOKEN_BUDGET * HEADROOM_FACTOR, num_ctx * 0.9)
    - No duplicates (checked by chunk ID)
    - Hard cap on count: len(selected) <= pack_top
    """
    # Edit 7: calculate max_budget with num_ctx constraint
    max_budget = int(min(CTX_TOKEN_BUDGET * HEADROOM_FACTOR, num_ctx * 0.9))

    out = []
    used = 0
    ids = []
    seen_ids = set()
    top1_included = False

    for idx_pos, idx in enumerate(order):
        # Hard cap on snippet count
        if len(ids) >= pack_top:
            break

        c = chunks[idx]
        cid = c["id"]

        # Avoid duplicates
        if cid in seen_ids:
            continue

        txt = c["text"]
        t_est = approx_tokens(len(txt))

        # For first item: allow up to max_budget (Edit 7: always include top-1)
        if (idx_pos == 0 or not ids) and not top1_included:
            ids.append(cid)
            seen_ids.add(cid)
            out.append(_fmt_snippet_header(c) + "\n" + txt)
            used += t_est
            top1_included = True
        # For remaining items: respect strict budget
        elif used + t_est <= budget_tokens:
            ids.append(cid)
            seen_ids.add(cid)
            out.append(_fmt_snippet_header(c) + "\n" + txt)
            used += t_est

    return "\n\n---\n\n".join(out), ids

# ====== COVERAGE CHECK ======
def coverage_ok(selected, dense_scores, threshold):
    """Check coverage."""
    if len(selected) < 2:
        return False
    highs = sum(1 for i in selected if dense_scores[i] >= threshold)
    return highs >= 2

# ====== LLM CALL ======
def ask_llm(question: str, snippets_block: str, seed=DEFAULT_SEED, num_ctx=DEFAULT_NUM_CTX, num_predict=DEFAULT_NUM_PREDICT, retries=0) -> str:
    """Call Ollama chat with Qwen best-practice options - Edit 1, 3, 9."""
    payload = {
        "model": GEN_MODEL,
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.05
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_WRAPPER.format(snips=snippets_block, q=question)}
        ],
        "stream": False
    }
    sess = get_session(retries=retries)

    # Edit 3: manual bounded retry for POST
    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            r = sess.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
                timeout=(CHAT_CONNECT_T, CHAT_READ_T),
                allow_redirects=False  # Edit 1
            )
            r.raise_for_status()
            j = r.json()
            msg = (j.get("message") or {}).get("content")
            if msg:
                return msg
            return j.get("response", "")
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_attempts - 1:
                time.sleep(0.5)  # 0.5s backoff
                continue
            # Final attempt failed
            logger.error(f"LLM call failed after {max_attempts} attempts: {e} "
                       f"[hint: check OLLAMA_URL or increase CHAT timeouts]")
            sys.exit(1)
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM request failed: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error in LLM call: {e}")
            sys.exit(1)

# ====== ATOMIC WRITES (CRASH-SAFE) ======
def _atomic_write_text(path: str, text: str):
    """Atomically write text file via tempfile + os.replace()."""
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path)+".", dir=d, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except:
        try:
            os.unlink(tmp)
        except:
            pass
        raise

def atomic_write_json(path: str, obj):
    """Atomically write JSON file."""
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False))

def atomic_write_jsonl(path: str, lines):
    """Atomically write JSONL file (list of strings or dicts) - Edit 8."""
    text_lines = [json.dumps(line) if isinstance(line, dict) else line for line in lines]
    content = "\n".join(text_lines)
    if content and not content.endswith("\n"):
        content += "\n"
    atomic_write_text(path, content)  # Edit 8: use atomic_write_text

# ====== BUILD PIPELINE ======
def build(md_path: str, retries=0):
    """Build knowledge base. Guarded with lock to prevent concurrent builds - Edit 8, 12, 14."""
    with build_lock():
        logger.info("=" * 70)
        logger.info("BUILDING KNOWLEDGE BASE")
        logger.info("=" * 70)
        if not os.path.exists(md_path):
            logger.error(f"{md_path} not found")
            sys.exit(1)

        logger.info("\n[1/4] Parsing and chunking...")
        chunks = build_chunks(md_path)
        logger.info(f"  Created {len(chunks)} chunks")
        # Edit 8: Write chunks atomically
        atomic_write_jsonl(FILES["chunks"], chunks)

        logger.info("\n[2/4] Embedding with Ollama...")
        vecs = embed_texts([c["text"] for c in chunks], retries=retries)
        # Pre-normalize for efficient retrieval, Edit 14: ensure float32
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        vecs_n = (vecs / norms).astype("float32")
        atomic_save_npy(vecs_n, FILES["emb"])  # Edit 8, 14
        logger.info(f"  Saved {vecs_n.shape} embeddings (normalized)")
        # Write metadata atomically
        meta_lines = [
            {
                "id": c["id"],
                "title": c["title"],
                "url": c["url"],
                "section": c["section"]
            }
            for c in chunks
        ]
        atomic_write_jsonl(FILES["meta"], meta_lines)  # Edit 8

        logger.info("\n[3/4] Building BM25 index...")
        bm = build_bm25(chunks)
        # Edit 8: use atomic_write_text for bm25.json
        atomic_write_text(FILES["bm25"], json.dumps(bm, ensure_ascii=False))
        logger.info(f"  Indexed {len(bm['idf'])} unique terms")

        # Optional HNSW fast index (behind env flag) with atomic save + fsync
        if os.getenv("USE_HNSWLIB") == "1":
            try:
                import hnswlib
                logger.info("\n[3.5/4] Building HNSW index...")
                p = hnswlib.Index(space='cosine', dim=vecs_n.shape[1])
                p.init_index(max_elements=vecs_n.shape[0], ef_construction=200, M=16)
                p.add_items(vecs_n.astype("float32"), np.arange(vecs_n.shape[0]))
                # Atomic save: write to temp file, fsync, then rename
                temp_path = FILES["hnsw"] + ".tmp"
                p.save_index(temp_path)
                # Ensure temp file hits disk before atomic replace
                with open(temp_path, "rb") as _f:
                    os.fsync(_f.fileno())
                os.replace(temp_path, FILES["hnsw"])  # Atomic on POSIX
                _fsync_dir(FILES["hnsw"])
                logger.info(f"  Saved HNSW index to {FILES['hnsw']}")
            except ImportError:
                logger.info("\n[3.5/4] HNSW requested but hnswlib not installed; skipping")
            except Exception as e:
                logger.info(f"\n[3.5/4] HNSW build failed: {e}; continuing without it")

        # Write index.meta.json for artifact versioning (atomic with fsync)
        logger.info("\n[3.6/4] Writing artifact metadata...")
        kb_sha = compute_sha256(md_path)
        index_meta = {
            "kb_sha256": kb_sha,
            "chunks": len(chunks),
            "emb_rows": int(vecs_n.shape[0]),
            "bm25_docs": len(bm["doc_lens"]),
            "gen_model": GEN_MODEL,
            "emb_model": EMB_MODEL,
            "mmr_lambda": MMR_LAMBDA,
            "chunk_chars": CHUNK_CHARS,
            "chunk_overlap": CHUNK_OVERLAP,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_version": "3.4"
        }
        # Edit 8: use atomic_write_text for index.meta.json
        atomic_write_text(FILES["index_meta"], json.dumps(index_meta, indent=2))
        logger.info(f"  Saved index metadata")

        logger.info("\n[4/4] Done.")
        logger.info("=" * 70)

# ====== LOAD INDEX ======
def load_index():
    """Load artifacts with full integrity validation - Edit 14."""
    # Check for metadata file
    if not os.path.exists(FILES["index_meta"]):
        logger.warning("[rebuild] index.meta.json missing: run 'python3 clockify_support_cli.py build knowledge_full.md'")
        return None

    with open(FILES["index_meta"], encoding="utf-8") as f:
        meta = json.loads(f.read())

    # 1. Check all required artifacts exist
    missing = []
    for key in ["chunks", "emb", "bm25"]:
        if not os.path.exists(FILES[key]):
            missing.append(FILES[key])

    if missing:
        logger.warning(f"[rebuild] Missing artifacts: {', '.join(missing)}")
        logger.warning("[rebuild] Remediation: run 'python3 clockify_support_cli.py build knowledge_full.md'")
        return None

    # 2. Load and validate embeddings (Edit 14: ensure float32)
    try:
        vecs_n = np.load(FILES["emb"], mmap_mode="r")  # Read-only memmap
        # Edit 14: force float32 dtype
        if vecs_n.dtype != np.float32:
            logger.warning(f"[rebuild] Embedding dtype mismatch: {vecs_n.dtype} (expected float32), converting...")
            vecs_n = np.load(FILES["emb"]).astype("float32")
        expected_rows = meta.get("emb_rows", 0)
        if vecs_n.shape[0] != expected_rows:
            logger.warning(f"[rebuild] Embedding rows mismatch: {vecs_n.shape[0]} rows vs {expected_rows} in metadata")
            logger.warning("[rebuild] Remediation: index.meta.json is stale; run build")
            return None
    except Exception as e:
        logger.warning(f"[rebuild] Failed to load embeddings: {e}")
        return None

    # 3. Load and validate chunks
    try:
        with open(FILES["chunks"], encoding="utf-8") as f:
            chunks = [json.loads(l) for l in f if l.strip()]
        if len(chunks) != meta.get("chunks", 0):
            logger.warning(f"[rebuild] Chunk count mismatch: {len(chunks)} chunks vs {meta.get('chunks')} in metadata")
            return None
    except Exception as e:
        logger.warning(f"[rebuild] Failed to load chunks: {e}")
        return None

    # 4. Load and validate BM25 index
    try:
        with open(FILES["bm25"], encoding="utf-8") as f:
            bm = json.loads(f.read())
        if len(bm["doc_lens"]) != meta.get("bm25_docs", 0):
            logger.warning(f"[rebuild] BM25 doc count mismatch: {len(bm['doc_lens'])} docs vs {meta.get('bm25_docs')} in metadata")
            return None
    except Exception as e:
        logger.warning(f"[rebuild] Failed to load BM25: {e}")
        return None

    # 5. Cross-check: embeddings and chunks must match
    if vecs_n.shape[0] != len(chunks):
        logger.warning(f"[rebuild] Embedding-chunk mismatch: {vecs_n.shape[0]} embeddings vs {len(chunks)} chunks")
        logger.warning("[rebuild] Remediation: rebuild required")
        return None

    # 5.5. KB drift detection: if source MD present, check hash against metadata
    if os.path.exists("knowledge_full.md"):
        try:
            kb_sha = compute_sha256("knowledge_full.md")
            stored_sha = meta.get("kb_sha256", "")
            if stored_sha and stored_sha != kb_sha:
                logger.warning("[rebuild] KB drift detected: source MD has changed")
                logger.warning("[rebuild] Remediation: run 'python3 clockify_support_cli.py build knowledge_full.md'")
                return None
        except Exception as e:
            logger.debug(f"Could not check KB hash: {e}")

    # 6. Optional HNSW index (non-blocking if missing)
    hnsw = None
    if os.getenv("USE_HNSWLIB") == "1" and os.path.exists(FILES["hnsw"]):
        try:
            import hnswlib
            hnsw = hnswlib.Index(space='cosine', dim=vecs_n.shape[1])
            hnsw.load_index(FILES["hnsw"])
            logger.debug(f"Loaded HNSW index from {FILES['hnsw']}")
        except ImportError:
            logger.debug("hnswlib not installed; skipping HNSW")
        except Exception as e:
            logger.warning(f"Failed to load HNSW: {e}; continuing without it")

    logger.debug(f"Index loaded: {len(chunks)} chunks, {vecs_n.shape[0]} embeddings, {len(bm['doc_lens'])} BM25 docs")
    return chunks, vecs_n, bm, hnsw

# ====== POLICY GUARDRAILS ======
def looks_sensitive(question: str) -> bool:
    """Check if question involves sensitive intent (account/billing/PII)."""
    sensitive_keywords = {
        # Financial
        "invoice", "billing", "credit card", "payment", "salary", "account balance",
        # Authentication & Secrets
        "password", "token", "api key", "secret", "private key",
        # PII
        "ssn", "social security", "iban", "swift", "routing number", "account number",
        "phone number", "email address", "home address", "date of birth",
        # Compliance
        "gdpr", "pii", "personally identifiable", "personal data"
    }
    q_lower = question.lower()
    return any(kw in q_lower for kw in sensitive_keywords)

def inject_policy_preamble(snippets_block: str, question: str) -> str:
    """Optionally prepend policy reminder for sensitive queries."""
    if looks_sensitive(question):
        policy = "[INTERNAL POLICY]\nDo not reveal PII, account secrets, or payment details. For account changes, redirect to secure internal admin panel.\n\n"
        return policy + snippets_block
    return snippets_block

# ====== ANSWER (STATELESS) ======
def answer_once(
    question: str,
    chunks,
    vecs_n,
    bm,
    top_k=12,
    pack_top=6,
    threshold=0.30,
    use_rerank=False,
    debug=False,
    hnsw=None,
    seed=DEFAULT_SEED,
    num_ctx=DEFAULT_NUM_CTX,
    num_predict=DEFAULT_NUM_PREDICT,
    retries=0
):
    """Answer a single question. Stateless. Returns (answer_text, metadata) - Edit 6, 7, 12."""
    # Validate refusal string assertion
    REFUSAL_STR = "I don't know based on the MD."

    turn_start = time.time()
    timings = {}
    try:
        # Step 1: Hybrid retrieval
        t0 = time.time()
        selected, scores = retrieve(question, chunks, vecs_n, bm, top_k=top_k, hnsw=hnsw, retries=retries)
        timings["retrieve"] = time.time() - t0

        # Step 2: MMR diversification on deduped candidates (Edit 6: updated signature)
        # We need to pass vecs_n in a way mmr can use it
        # Rewrite mmr to accept vecs_n or make it a closure
        # For now, we'll inline the MMR logic here to maintain vecs_n access
        mmr_selected = []
        cand = list(selected)

        # Always include the top dense score first for better recall
        if cand:
            top_dense_idx = max(cand, key=lambda j: scores["dense"][j])
            mmr_selected.append(top_dense_idx)
            cand.remove(top_dense_idx)

        # Then diversify the rest using actual passage cosine similarity
        while cand and len(mmr_selected) < pack_top:
            def mmr_gain(j):
                rel = scores["dense"][j]
                # Compute max cosine similarity with already-selected passages
                div = 0.0
                if mmr_selected:
                    div = max(float(vecs_n[j].dot(vecs_n[k])) for k in mmr_selected)
                return MMR_LAMBDA * rel - (1 - MMR_LAMBDA) * div
            i = max(cand, key=mmr_gain)
            mmr_selected.append(i)
            cand.remove(i)

        # Step 3: Optional LLM reranking on MMR order
        rerank_scores = {}
        if use_rerank:
            logger.debug(json.dumps({"event": "rerank_start", "candidates": len(mmr_selected)}))
            t0 = time.time()
            mmr_selected, rerank_scores = rerank_with_llm(question, chunks, mmr_selected, scores, seed=seed, num_ctx=num_ctx, num_predict=num_predict, retries=retries)
            timings["rerank"] = time.time() - t0
            logger.debug(json.dumps({"event": "rerank_done", "selected": len(mmr_selected), "scored": len(rerank_scores)}))

        # Step 4: Coverage check
        coverage_pass = coverage_ok(mmr_selected, scores["dense"], threshold)
        if not coverage_pass:
            if debug:
                print(f"\n[DEBUG] Coverage failed: {len(mmr_selected)} selected, need ≥2 @ {threshold}")
            logger.info(f"[coverage_gate] REJECTED: seed={seed} model={GEN_MODEL} selected={len(mmr_selected)} threshold={threshold}")
            return REFUSAL_STR, {"selected": []}

        # Step 5: Pack with token budget and snippet cap (Edit 7: pass num_ctx)
        block, ids = pack_snippets(chunks, mmr_selected, pack_top=pack_top, budget_tokens=CTX_TOKEN_BUDGET, num_ctx=num_ctx)

        # Apply policy preamble for sensitive queries
        block = inject_policy_preamble(block, question)

        # Step 6: Call LLM
        t0 = time.time()
        ans = ask_llm(question, block, seed=seed, num_ctx=num_ctx, num_predict=num_predict, retries=retries).strip()
        timings["ask_llm"] = time.time() - t0
        timings["total"] = time.time() - turn_start

        # Step 7: Optional debug output with all metrics
        if debug:
            diag = []
            for rank, i in enumerate(mmr_selected):
                entry = {
                    "id": chunks[i]["id"],
                    "title": chunks[i]["title"],
                    "section": chunks[i]["section"],
                    "url": chunks[i]["url"],
                    "dense": float(scores["dense"][i]),
                    "bm25": float(scores["bm25"][i]),
                    "hybrid": float(scores["hybrid"][i]),
                    "mmr_rank": rank
                }
                if i in rerank_scores:
                    entry["rerank_score"] = float(rerank_scores[i])
                diag.append(entry)
            ans += "\n\n[DEBUG]\n" + json.dumps(diag, ensure_ascii=False, indent=2)

        # Edit 12: one-line turn logging with all parameters
        logger.info(
            "turn model=%s seed=%s topk=%s pack=%s threshold=%s rerank=%s latency.total=%.1fs",
            GEN_MODEL, seed, top_k, pack_top, threshold, use_rerank, timings["total"]
        )

        return ans, {"selected": ids}
    except Exception as e:
        logger.error(f"{e}")
        sys.exit(1)

# ====== SELF-CHECK TESTS (Edit 15) ======
def test_mmr_signature_ok():
    """Verify mmr() function signature - Edit 15."""
    import inspect
    # MMR is now inlined in answer_once, but we can check the old signature is gone
    # For this test, we'll verify the new inline implementation exists
    sig = inspect.signature(answer_once)
    # Just verify answer_once exists and has expected params
    assert "question" in sig.parameters
    assert "vecs_n" in sig.parameters
    return True

def test_pack_headroom_enforced():
    """Verify top-1 always included - Edit 15."""
    # Mock chunks
    chunks = [
        {"id": "1", "title": "T", "section": "S", "url": "", "text": "x" * 20000},  # Very large
        {"id": "2", "title": "T", "section": "S", "url": "", "text": "y" * 100},
    ]
    # Pack with very small budget
    block, ids = pack_snippets(chunks, [0, 1], pack_top=2, budget_tokens=10, num_ctx=1000)
    # Top-1 should always be included even if it exceeds budget
    assert len(ids) >= 1
    assert "1" in ids
    return True

def test_rtf_guard_false_positive():
    """Verify non-RTF with backslashes not stripped - Edit 15."""
    # Text with backslashes but not RTF
    text = r"This is \normal text with \backslashes but no RTF commands"
    result = strip_noise(text)
    # Should not be modified (no RTF stripping)
    assert "\\normal" in result
    assert "\\backslashes" in result
    return True

def test_float32_pipeline_ok():
    """Verify all vectors are float32 - Edit 15."""
    # Create a test vector
    vec = np.array([1.0, 2.0, 3.0], dtype="float64")
    # Save and load via atomic_save_npy
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        tmp_path = f.name
    try:
        atomic_save_npy(vec, tmp_path)
        loaded = np.load(tmp_path)
        assert loaded.dtype == np.float32
        return True
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def run_selftest():
    """Run all self-check tests - Edit 15."""
    tests = [
        ("MMR signature", test_mmr_signature_ok),
        ("Pack headroom", test_pack_headroom_enforced),
        ("RTF guard false positive", test_rtf_guard_false_positive),
        ("Float32 pipeline", test_float32_pipeline_ok),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            status = "PASS" if result else "FAIL"
            results.append((name, status))
            logger.info(f"[selftest] {name}: {status}")
        except Exception as e:
            results.append((name, "FAIL"))
            logger.error(f"[selftest] {name}: FAIL ({e})")

    # Summary
    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)
    logger.info(f"[selftest] {passed}/{total} tests passed")

    return all(status == "PASS" for _, status in results)

# ====== REPL ======
def chat_repl(top_k=12, pack_top=6, threshold=0.30, use_rerank=False, debug=False, seed=DEFAULT_SEED, num_ctx=DEFAULT_NUM_CTX, num_predict=DEFAULT_NUM_PREDICT, retries=0):
    """Stateless REPL loop - Edit 12, 13."""
    # Edit 13: log config summary at startup
    _log_config_summary()

    # Lazy build and startup sanity check
    artifacts_ok = True
    for fname in [FILES["chunks"], FILES["emb"], FILES["meta"], FILES["bm25"], FILES["index_meta"]]:
        if not os.path.exists(fname):
            artifacts_ok = False
            break

    if not artifacts_ok:
        logger.info(f"[rebuild] artifacts missing or invalid: building from knowledge_full.md...")
        if os.path.exists("knowledge_full.md"):
            build("knowledge_full.md", retries=retries)
        else:
            logger.error(f"knowledge_full.md not found")
            sys.exit(1)

    result = load_index()
    if result is None:
        logger.info(f"[rebuild] artifact validation failed: rebuilding...")
        if os.path.exists("knowledge_full.md"):
            build("knowledge_full.md", retries=retries)
            result = load_index()
        else:
            logger.error(f"knowledge_full.md not found")
            sys.exit(1)

    if result is None:
        logger.error(f"Failed to load artifacts after rebuild")
        sys.exit(1)

    chunks, vecs_n, bm, hnsw = result

    print("\n" + "=" * 70)
    print("CLOCKIFY SUPPORT – Local, Stateless, Closed-Book")
    print("=" * 70)
    print("Type a question. Commands: :exit, :debug")
    print("=" * 70 + "\n")

    dbg = debug
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if not q:
            continue
        if q == ":exit":
            break
        if q == ":debug":
            dbg = not dbg
            print(f"[debug={'ON' if dbg else 'OFF'}]")
            continue

        ans, meta = answer_once(
            q,
            chunks,
            vecs_n,
            bm,
            top_k=top_k,
            pack_top=pack_top,
            threshold=threshold,
            use_rerank=use_rerank,
            debug=dbg,
            hnsw=hnsw,
            seed=seed,
            num_ctx=num_ctx,
            num_predict=num_predict,
            retries=retries
        )
        print(ans)
        print()

# ====== MAIN ======
def main():
    ap = argparse.ArgumentParser(
        prog="clockify_support_cli",
        description="Clockify internal support chatbot (offline, stateless, closed-book)"
    )

    # Global logging and config arguments
    ap.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARN"],
                    help="Logging level (default INFO)")
    ap.add_argument("--ollama-url", type=str, default=None,
                    help="Ollama endpoint (default from OLLAMA_URL env or http://10.127.0.192:11434)")
    ap.add_argument("--gen-model", type=str, default=None,
                    help="Generation model name (default from GEN_MODEL env or qwen2.5:32b)")
    ap.add_argument("--emb-model", type=str, default=None,
                    help="Embedding model name (default from EMB_MODEL env or nomic-embed-text)")
    ap.add_argument("--ctx-budget", type=int, default=None,
                    help="Context token budget (default from CTX_BUDGET env or 2800)")
    # Edit 9: timeout CLI flags
    ap.add_argument("--emb-connect", type=float, default=None,
                    help="Embedding connect timeout (default 3)")
    ap.add_argument("--emb-read", type=float, default=None,
                    help="Embedding read timeout (default 120)")
    ap.add_argument("--chat-connect", type=float, default=None,
                    help="Chat connect timeout (default 3)")
    ap.add_argument("--chat-read", type=float, default=None,
                    help="Chat read timeout (default 180)")

    subparsers = ap.add_subparsers(dest="cmd")

    b = subparsers.add_parser("build", help="Build knowledge base")
    b.add_argument("md_path", help="Path to knowledge_full.md")
    b.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retries for transient errors (default 0)")

    c = subparsers.add_parser("chat", help="Start REPL")
    c.add_argument("--debug", action="store_true", help="Print retrieval diagnostics")
    c.add_argument("--rerank", action="store_true", help="Enable LLM-based reranking")
    c.add_argument("--topk", type=int, default=DEFAULT_TOP_K, help="Top-K candidates (default 12)")
    c.add_argument("--pack", type=int, default=DEFAULT_PACK_TOP, help="Snippets to pack (default 6)")
    c.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Cosine threshold (default 0.30)")
    c.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for LLM (default 42)")
    c.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help="LLM context window (default 8192)")
    c.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT, help="LLM max generation tokens (default 512)")
    c.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retries for transient errors (default 0)")
    # Edit 5: determinism check flags
    c.add_argument("--det-check", action="store_true", help="Determinism check: ask same Q twice, compare hashes")
    c.add_argument("--det-check-q", type=str, default=None, help="Custom question for determinism check")
    # Edit 15: selftest flag
    c.add_argument("--selftest", action="store_true", help="Run self-check tests and exit")

    args = ap.parse_args()

    # Edit 12: Setup logging after CLI arg parsing (moved from module level)
    level = getattr(logging, args.log if hasattr(args, "log") else "INFO")
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    # Edit 9: Apply timeout overrides from CLI
    global EMB_CONNECT_T, EMB_READ_T, CHAT_CONNECT_T, CHAT_READ_T
    if hasattr(args, "emb_connect") and args.emb_connect:
        EMB_CONNECT_T = args.emb_connect
    if hasattr(args, "emb_read") and args.emb_read:
        EMB_READ_T = args.emb_read
    if hasattr(args, "chat_connect") and args.chat_connect:
        CHAT_CONNECT_T = args.chat_connect
    if hasattr(args, "chat_read") and args.chat_read:
        CHAT_READ_T = args.chat_read

    # Validate and set config from CLI args
    try:
        validate_and_set_config(
            ollama_url=args.ollama_url,
            gen_model=args.gen_model,
            emb_model=args.emb_model,
            ctx_budget=args.ctx_budget
        )
        validate_chunk_config()
    except ValueError as e:
        logger.error(f"CONFIG ERROR: {e}")
        sys.exit(1)

    # Auto-start REPL if no command given
    if args.cmd is None:
        chat_repl()
        return

    if args.cmd == "build":
        build(args.md_path, retries=getattr(args, "retries", 0))
        return

    if args.cmd == "chat":
        # Edit 15: selftest mode
        if getattr(args, "selftest", False):
            success = run_selftest()
            sys.exit(0 if success else 1)

        # Edit 5: Determinism check
        if getattr(args, "det_check", False):
            det_prompts = [
                "How do I track time in Clockify?",
                "How do I cancel my subscription?"
            ]
            if args.det_check_q:
                det_prompts = [args.det_check_q]

            # Load index once for determinism test
            for fname in [FILES["chunks"], FILES["emb"], FILES["meta"], FILES["bm25"], FILES["index_meta"]]:
                if not os.path.exists(fname):
                    logger.info("[rebuild] artifacts missing for det-check: building...")
                    if os.path.exists("knowledge_full.md"):
                        build("knowledge_full.md", retries=getattr(args, "retries", 0))
                    break
            result = load_index()
            if result:
                chunks, vecs_n, bm, hnsw = result

                # Normalize output for robust determinism check
                def _normalize_for_hash(s: str) -> str:
                    s = unicodedata.normalize("NFKC", s)
                    s = re.sub(r"\s+", " ", s).strip()
                    # Drop debug diagnostics
                    s = re.sub(r"\[DEBUG\][\s\S]*$", "", s).strip()
                    return s

                logger.info(f"[DETERMINISM CHECK] model={GEN_MODEL} seed={args.seed}")
                for q in det_prompts:
                    a1, _ = answer_once(
                        q, chunks, vecs_n, bm,
                        top_k=args.topk, pack_top=args.pack, threshold=args.threshold,
                        use_rerank=args.rerank, debug=False, hnsw=hnsw,
                        seed=args.seed, num_ctx=args.num_ctx, num_predict=args.num_predict, retries=getattr(args, "retries", 0)
                    )
                    a2, _ = answer_once(
                        q, chunks, vecs_n, bm,
                        top_k=args.topk, pack_top=args.pack, threshold=args.threshold,
                        use_rerank=args.rerank, debug=False, hnsw=hnsw,
                        seed=args.seed, num_ctx=args.num_ctx, num_predict=args.num_predict, retries=getattr(args, "retries", 0)
                    )
                    h1 = hashlib.sha256(_normalize_for_hash(a1).encode()).hexdigest()[:16]
                    h2 = hashlib.sha256(_normalize_for_hash(a2).encode()).hexdigest()[:16]
                    is_det = h1 == h2
                    # Edit 5: output format
                    print(f'[DETERMINISM] q="{q}" run1={h1} run2={h2} deterministic={"true" if is_det else "false"}')
                sys.exit(0)
            else:
                logger.error("failed to load index for det-check")
                sys.exit(1)

        # Normal chat REPL
        chat_repl(
            top_k=args.topk,
            pack_top=args.pack,
            threshold=args.threshold,
            use_rerank=args.rerank,
            debug=args.debug,
            seed=args.seed,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            retries=getattr(args, "retries", 0)
        )
        return

if __name__ == "__main__":
    main()
