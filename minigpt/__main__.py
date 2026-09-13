"""``python -m minigpt`` as an alias for ``python -m minigpt.cli``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
