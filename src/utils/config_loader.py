"""Config Loader — wczytywanie plików YAML do dataclass/dict."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_configs(configs_dir: Path = Path("configs")) -> dict[str, Any]:
    """Funkcja wczytuje wszystkie YAML z katalogu configs/ i scala w jeden słownik."""
    combined: dict[str, Any] = {}
    for yaml_file in sorted(configs_dir.glob("*.yaml")):
        data = load_config(yaml_file)
        if data:
            combined[yaml_file.stem] = data
    return combined
