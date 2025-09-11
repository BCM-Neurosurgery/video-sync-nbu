from pathlib import Path


def _name(p: str | Path) -> str:
    """basename with extension (e.g., 'file.csv')."""
    try:
        return Path(p).name
    except Exception:
        return str(p)
