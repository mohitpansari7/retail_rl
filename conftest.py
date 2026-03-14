"""
conftest.py
────────────
Pytest configuration file.

WHY this file exists:
  pytest needs to know where the project root is so that
  `from agents.reinforce_agent import ReinforceAgent` resolves correctly
  when you run `pytest tests/` from the retail_rl/ directory.

  Placing conftest.py at the project root tells pytest:
  "this directory is the root — add it to sys.path automatically."

  Without this, you'd need to run:
    PYTHONPATH=. pytest tests/
  every time. With this file, just:
    pytest tests/
"""
import sys
import os

# Add project root to path so all `from config.x import y` style
# imports work regardless of where pytest is invoked from.
sys.path.insert(0, os.path.dirname(__file__))