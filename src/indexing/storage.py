"""Utilities for persisting indexing results."""
import json
import os


def save_json(data, output_path, ensure_ascii=True):
    """Save dict to JSON, creating parent directories if needed."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=2)
