"""Allow running asub as ``python -m asub``."""

import sys

from asub.cli import main

sys.exit(main())
