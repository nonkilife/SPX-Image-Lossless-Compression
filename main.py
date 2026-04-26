"""Thin shim — delegates to spx.__main__ for backward compatibility."""
from spx.__main__ import main

if __name__ == "__main__":
    main()
