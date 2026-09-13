"""Present so that ``pytest`` finds the ``minigpt`` package from the repository root.

pytest inserts the directory containing the topmost ``conftest.py`` at the front of ``sys.path``,
which makes ``import minigpt`` work whether the suite is run as ``pytest``, ``pytest tests/``, or
``python -m pytest`` from anywhere in the tree. Without it, the import succeeds only when the
working directory happens to be the repository root -- a failure that looks like a broken package
and is really a path problem.
"""
