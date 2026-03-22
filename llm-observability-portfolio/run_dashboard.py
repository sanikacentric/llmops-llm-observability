#!/usr/bin/env python3
"""
run_dashboard.py
────────────────
Starts the LLM Observability Dashboard on http://localhost:8080

Usage:
    python run_dashboard.py
"""

import sys
from pathlib import Path

# Load .env before anything else
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    import uvicorn
    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  LLM Observability Dashboard                    │")
    print("  │  http://localhost:8080                          │")
    print("  │                                                 │")
    print("  │  1. Open http://localhost:8080 in your browser  │")
    print("  │  2. Click 'Run All Projects' to generate data   │")
    print("  │  3. Watch all 4 projects run live (~60s)        │")
    print("  └─────────────────────────────────────────────────┘")
    print()
    uvicorn.run(
        "dashboard.server:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
        reload=False,
    )
