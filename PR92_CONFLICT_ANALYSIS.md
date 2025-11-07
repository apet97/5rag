# PR #92 Merge Conflict Analysis

## Overview

PR #92 ("Refactor runtime config resolution for RAG components") has merge conflicts with the main branch. The conflict stems from **fundamentally different architectural approaches** that were developed in parallel.

## Conflict Location

**File:** `clockify_rag/retrieval.py`
**Lines:** 42-110 (prompts section) and throughout (config access pattern)

---

## What Changed in Each Branch

### PR #92 (codex/refactor-clockify_rag-for-config-usage)

**Architectural Changes:**
1. **Config Access Pattern:** Uses `import clockify_rag.config as config` with namespace access (`config.CONSTANT`)
2. **Dynamic Prompts:** Introduces `get_system_prompt()` function for runtime prompt generation
3. **Template-based Prompts:** Uses `_SYSTEM_PROMPT_TEMPLATE` with `.format(refusal=config.REFUSAL_STR)`
4. **Detailed USER_WRAPPER:** Includes explicit JSON structure examples and confidence scoring guidelines
5. **Optional Parameters:** Functions accept `Optional[int] = None` and resolve internally

**Example:**
```python
import clockify_rag.config as config

_SYSTEM_PROMPT_TEMPLATE = """... reply exactly "{refusal}" ..."""

def get_system_prompt() -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(refusal=config.REFUSAL_STR)

def ask_llm(seed: Optional[int] = None, ...):
    if seed is None:
        seed = config.DEFAULT_SEED
```

**Commits (7 total):**
- 76f25fc: Document dynamic retrieval prompt accessor
- 9ebf84c: Expose dynamic system prompt via module getattr
- 88d171a: Add tests for dynamic system prompt config overrides
- e66dafa: Align system prompt with JSON response guidelines
- 616f7d2: Refactor runtime config usage and add regression tests

---

### Main Branch (via PR #91)

**Architectural Changes:**
1. **Config Access Pattern:** Direct imports from config module (`from .config import REFUSAL_STR, ...`)
2. **Static Prompts:** Simple f-string evaluated at module load time
3. **Simplified USER_WRAPPER:** Minimal, concise instructions
4. **Direct Defaults:** Functions use direct constant references in default parameters

**Example:**
```python
from .config import REFUSAL_STR, DEFAULT_SEED, ...

SYSTEM_PROMPT = f"""... reply exactly "{REFUSAL_STR}" ..."""

def ask_llm(seed=DEFAULT_SEED, ...):
    # Use seed directly
```

**Commits (from PR #91):**
- 3392c26: Refine JSON prompts and extend answer parsing tests
- 11ea0ed: Merge PR #91

---

## Key Differences

| Aspect | PR #92 | Main Branch |
|--------|---------|-------------|
| **Config Access** | `config.CONSTANT` | Direct import |
| **Prompt Type** | Template + function | Static f-string |
| **Runtime Flexibility** | ✅ Dynamic (can change REFUSAL_STR at runtime) | ❌ Static (fixed at import) |
| **Complexity** | Higher (template system) | Lower (simple strings) |
| **Prompt Detail** | Verbose (explicit JSON examples) | Concise (minimal instructions) |
| **Function Signatures** | `Optional[int] = None` | `int = CONSTANT` |
| **Testability** | Better (can override config in tests) | Limited (module-level constants) |

---

## The Core Conflict

**Prompts Section (lines 42-110):**

```diff
# PR #92 has:
-_SYSTEM_PROMPT_TEMPLATE = """... "{refusal}" ..."""
-def get_system_prompt() -> str:
-    return _SYSTEM_PROMPT_TEMPLATE.format(refusal=config.REFUSAL_STR)
-USER_WRAPPER = """... [detailed JSON instructions] ..."""

# Main has:
+SYSTEM_PROMPT = f"""... "{REFUSAL_STR}" ..."""
+(no get_system_prompt function)
+USER_WRAPPER = """... [concise instructions] ..."""
```

**Config Usage Throughout:**
- PR #92: `config.ALPHA_HYBRID`, `config.GEN_MODEL`, etc.
- Main: `ALPHA_HYBRID`, `GEN_MODEL`, etc. (direct)

---

## Impact Assessment

### If PR #92 Merges As-Is (After Resolving Conflicts)

**Pros:**
- ✅ Runtime config flexibility (good for testing)
- ✅ Can dynamically change REFUSAL_STR without module reload
- ✅ Consistent config access pattern (`config.X` everywhere)
- ✅ Better separation of concerns (config in one place)

**Cons:**
- ❌ Requires resolving conflicts manually
- ❌ More complex than current main
- ❌ Regression tests may conflict with PR #91's test changes
- ❌ Verbose prompts may reduce LLM performance (more tokens to process)

### If Main's Approach Continues

**Pros:**
- ✅ Simpler, more Pythonic (direct imports)
- ✅ Concise prompts (PR #91's improvements)
- ✅ No merge conflicts (already in main)
- ✅ Faster module load (no runtime formatting)

**Cons:**
- ❌ Static config (harder to test with different values)
- ❌ Inconsistent config access (some use imports, some use `config.X`)
- ❌ No runtime prompt customization

---

## Recommended Resolution Strategy

### Option 1: **Adopt Main's Simplified Prompts with PR #92's Config Pattern** (RECOMMENDED)

**Approach:**
1. Keep PR #92's `import clockify_rag.config as config` pattern
2. **Replace** verbose prompts with main's simplified versions
3. Keep dynamic `get_system_prompt()` for testability
4. Update function signatures to use `Optional[int] = None` with internal resolution

**Rationale:**
- Best of both worlds: runtime flexibility + concise prompts
- PR #91's improvements show that verbose JSON instructions aren't needed
- Maintains PR #92's testability improvements
- Consistent config access pattern

**Work Required:**
- Medium effort
- Need to carefully merge prompt changes
- Update tests to match both changes

---

### Option 2: **Abandon PR #92, Align with Main**

**Approach:**
1. Close PR #92
2. Cherry-pick only the test improvements (if any)
3. Continue with main's simpler approach

**Rationale:**
- Fastest path forward
- Main already has PR #91's improvements
- Avoids complexity of dynamic prompts

**Work Required:**
- Low effort
- May lose some testability improvements

---

### Option 3: **Merge PR #92, Keep Verbose Prompts**

**Approach:**
1. Manually resolve conflicts, keeping PR #92's verbose prompts
2. Ignore PR #91's simplifications

**Rationale:**
- Preserves PR #92's original vision
- Maximum runtime flexibility

**Work Required:**
- High effort
- Need to resolve conflicts carefully
- May revert PR #91's improvements (not recommended)

---

## Detailed Conflict Resolution Steps (Option 1 - RECOMMENDED)

### Step 1: Prepare Branch
```bash
git checkout codex/refactor-clockify_rag-for-config-usage
git fetch origin
```

### Step 2: Manual Merge with Strategy

For `clockify_rag/retrieval.py`:

1. **Keep PR #92's imports:**
   ```python
   import clockify_rag.config as config
   ```

2. **Use Main's simplified SYSTEM_PROMPT but make it dynamic:**
   ```python
   _SYSTEM_PROMPT_TEMPLATE = """You are CAKE.com Internal Support for Clockify.
   Closed-book. Only use SNIPPETS. If info is missing, reply exactly "{refusal}" and set confidence to 0.
   Respond with a single JSON object that matches this schema:
   {{
     "answer": "<complete response>",
     "confidence": <0-100 integer>
   }}
   Guidelines for the answer field:
   - Use the user's language.
   - Be precise. No speculation. No external info. No web search.
   - Include the following sections in order inside the answer text (you may format them with numbered or bulleted lists):
     1. Direct answer.
     2. Steps.
     3. Notes by role/plan/region if relevant.
     4. Citations with snippet IDs like [id1, id2], including URLs inline if present.
   - If SNIPPETS disagree, explain the conflict and provide the safest interpretation.
   - Ensure the entire output remains valid JSON with no extra prose or markdown wrappers."""

   def get_system_prompt() -> str:
       """Return the system prompt with the current refusal string."""
       return _SYSTEM_PROMPT_TEMPLATE.format(refusal=config.REFUSAL_STR)
   ```

3. **Use Main's simplified USER_WRAPPER:**
   ```python
   USER_WRAPPER = """SNIPPETS:
   {snips}

   QUESTION:
   {q}

   Respond with only a JSON object following the schema {{"answer": "...", "confidence": 0-100}}.
   Keep all narrative content inside the answer field and include citations as described in the system message.
   Do not add markdown fences or text outside the JSON object."""
   ```

4. **Keep PR #92's function signatures with Optional parameters**

5. **Replace all direct constant usage with `config.X`:**
   - `ALPHA_HYBRID` → `config.ALPHA_HYBRID`
   - `GEN_MODEL` → `config.GEN_MODEL`
   - etc.

### Step 3: Test
```bash
pytest tests/test_runtime_config.py tests/test_retrieval.py tests/test_answer.py -v
```

### Step 4: Commit
```bash
git add clockify_rag/retrieval.py
git commit -m "Resolve merge conflict: adopt main's simplified prompts with dynamic config access"
```

---

## Files Affected by Merge

```bash
# Successfully merged (no conflicts):
- clockify_support_cli_final.py
- tests/test_answer.py

# Conflicted (need manual resolution):
- clockify_rag/retrieval.py
```

---

## Testing Checklist

After resolving conflicts, verify:

- [ ] `pytest tests/test_runtime_config.py` (PR #92's new tests)
- [ ] `pytest tests/test_retrieval.py` (existing retrieval tests)
- [ ] `pytest tests/test_answer.py` (PR #91's JSON parsing tests)
- [ ] `pytest tests/test_packer.py` (snippet packing tests)
- [ ] Manual smoke test: `python3 clockify_support_cli_final.py chat`
- [ ] Verify config mutations work at runtime (key feature of PR #92)

---

## Decision Points

**You need to decide:**

1. **Which config access pattern?**
   - [ ] PR #92's `import config as config` (better for testing)
   - [ ] Main's direct imports (simpler, more Pythonic)

2. **Which prompt style?**
   - [ ] Main's concise prompts (recommended based on PR #91 results)
   - [ ] PR #92's verbose prompts

3. **Dynamic or static prompts?**
   - [ ] Dynamic `get_system_prompt()` (runtime flexibility)
   - [ ] Static `SYSTEM_PROMPT` f-string (simpler)

**My Recommendation:** Use **Option 1** above - combine PR #92's config pattern with main's simplified prompts. This gives you runtime flexibility for testing while keeping the improvements from PR #91.

---

## Next Steps

1. Review this analysis
2. Choose resolution strategy (Option 1, 2, or 3)
3. I can help implement the chosen strategy
4. Run tests
5. Commit and push to the PR branch

**Would you like me to proceed with Option 1 (recommended) and resolve the conflicts for you?**
