import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def config_path() -> pathlib.Path:
    return ROOT / "config" / "default.yaml"
