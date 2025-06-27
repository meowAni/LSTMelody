import json
import os

class Config:
    _config = None

    @classmethod
    def _load_config(cls):
        if cls._config is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
            with open(config_path, "r") as f:
                cls._config = json.load(f)

    @classmethod
    def get(cls, key, default=None):
        cls._load_config()
        return cls._config.get(key, default)