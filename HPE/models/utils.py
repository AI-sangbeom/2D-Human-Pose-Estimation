from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

def is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in ("https", "file")

def convert_path_or_url_to_url(path: str) -> str:
    if is_url(path):
        return path
    return Path(path).expanduser().resolve().as_uri()

class Weights(Enum):
    LVD1689M = "LVD1689M"
    SAT493M = "SAT493M"
