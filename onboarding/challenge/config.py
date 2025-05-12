from pathlib import Path

import yaml


def load_settings(
    path: str = "challenge/settings.yaml",
) -> dict:
    settings_path = Path(__file__).resolve().parent.parent / path
    with settings_path.open("r") as f:
        return yaml.safe_load(f)


settings = load_settings()
