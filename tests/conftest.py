# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Makes the project root importable for the test suite.

The tests import the pipeline as ``from src... import ...``, which requires the project
root on ``sys.path``. ``python -m pytest`` puts it there implicitly, but the ``pytest``
console script does not: it prepends the test file's own directory (``tests/``), so
``import src`` fails. pytest always loads this file before importing the tests beside
it, so adding the root here fixes every invocation style -- ``pytest -q`` from the
project root, ``python -m pytest``, or an absolute path to ``tests/`` from any working
directory -- on the HPC server as well as locally.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
