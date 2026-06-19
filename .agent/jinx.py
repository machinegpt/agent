#!/usr/bin/env python3
"""
Direct entry point for JINX.
Designed to be executed when the repository is dropped into a project as a `.agent` folder.

Usage:
  python .agent/jinx.py "Analyze and fix the bug in auth"
"""
import sys
from pathlib import Path

# Insert the local src/ directory into sys.path so 'import jinx' resolves correctly
# without needing a global pip installation.
src_path = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(src_path))

def bootstrap_dependencies():
    try:
        import pydantic
        import yaml
    except ImportError:
        print("[JINX BOOTSTRAP] Missing dependencies. Installing pydantic and pyyaml...", file=sys.stderr)
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pydantic>=2.0.0", "pyyaml>=6.0"], stdout=subprocess.DEVNULL)
            print("[JINX BOOTSTRAP] Dependencies installed successfully.", file=sys.stderr)
        except Exception as e:
            print(f"[JINX BOOTSTRAP ERROR] Automatic dependency installation failed: {e}", file=sys.stderr)
            print("Please manually run: pip install pydantic>=2.0.0 pyyaml>=6.0", file=sys.stderr)
            sys.exit(1)

# Check and install dependencies before importing JINX package modules
bootstrap_dependencies()

try:
    from jinx.cli import main
except ImportError as e:
    print(f"[JINX BOOTSTRAP ERROR] Failed to load JINX modules: {e}", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()
