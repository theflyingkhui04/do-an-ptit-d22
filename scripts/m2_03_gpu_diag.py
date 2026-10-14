"""M2-03 targeted terminal stages; requires reviewed commit before GPU stages."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.m2_03_targeted.execution import main

if __name__ == "__main__":
    raise SystemExit(main())
