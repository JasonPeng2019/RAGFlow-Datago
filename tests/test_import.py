import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import datago

def test_import():
    print("datago imported, version=", datago.__version__)

if __name__ == "__main__":
    test_import()
