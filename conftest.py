"""Root conftest: ensure the project root is importable during tests.

Having an (empty) conftest.py at the repository root makes pytest add this
directory to ``sys.path``, so the tests can import ``app``, ``detector`` and
``config`` as top-level modules.
"""
