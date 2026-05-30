# -*- coding: utf-8 -*-
"""Compatibility launcher for the UPW RL system.

Preferred entry point:
    python -m upw_rl_system --csv path/to/data.csv
"""

from upw_rl_system.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
