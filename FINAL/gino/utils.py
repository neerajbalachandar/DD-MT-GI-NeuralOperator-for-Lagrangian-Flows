import json
from pathlib import Path
import yaml


def load_config(path, overrides=()):
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for item in overrides:
        key, raw = item.split("=", 1)
        import yaml as parser
        value = parser.safe_load(raw)
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return config


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
