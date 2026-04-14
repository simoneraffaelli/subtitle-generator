"""Allow running subtitler as ``python -m subtitler``."""

import sys

from subtitler.cli import main

sys.exit(main())
