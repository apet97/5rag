#!/usr/bin/env python3
"""Backward compatibility wrapper for clockify_support_cli_final.py

This wrapper maintains the original CLI interface while using the new
modular clockify_rag package structure. This allows existing scripts and
workflows to continue working without modification.

For new code, prefer importing from clockify_rag directly:
    from clockify_rag import build, load_index
"""

# Import all functionality from the final implementation
# This ensures backward compatibility
from clockify_support_cli_final import *

if __name__ == "__main__":
    main()
