import sys
from pathlib import Path

# Make the repo root importable so `from defect_demo.run import ...` works
# without needing to install the project as a package.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
