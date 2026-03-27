import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

# VITRA repository root (…/vitra/utils/config_utils.py → parents[2])
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def deep_update(d1, d2):
    """Deep update d1 with d2, recursively merging nested dictionaries."""
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1:
            assert isinstance(d1[k], dict), f"Cannot merge dict with non-dict for key {k}"
            deep_update(d1[k], d2[k])
        else:
            d1[k] = d2[k]
    return d1


def resolve_repo_paths(config: dict) -> None:
    """Apply after CLI overrides so paths stay absolute relative to the repo root."""
    _resolve_config_paths(config, REPO_ROOT)


def _resolve_config_paths(config: dict, root: Path) -> None:
    """Turn repo-relative paths under `weights/` into absolute paths so inference uses local files."""

    def maybe_resolve(p: str) -> str:
        if not isinstance(p, str) or not p:
            return p
        if os.path.isabs(p):
            return p
        candidate = (root / p).resolve()
        if candidate.exists():
            return str(candidate)
        return p

    for key in ("model_load_path", "statistics_path"):
        if key in config and config[key]:
            config[key] = maybe_resolve(config[key])
    vlm = config.get("vlm")
    if isinstance(vlm, dict):
        pm = vlm.get("pretrained_model_name_or_path")
        if pm:
            vlm["pretrained_model_name_or_path"] = maybe_resolve(pm)


def load_config(config_file):
    """Load configuration file with support for parent configs and Hugging Face Hub."""
    # Check if config_file is a Hugging Face repo (format: "username/repo-name:filename")
    from_huggingface = False
    if ":" in config_file and "/" in config_file.split(":")[0] and not os.path.exists(config_file):
        # Parse Hugging Face repo format: "username/repo-name:config.json"
        repo_id, filename = config_file.split(":", 1)
        print(f"Loading config from Hugging Face Hub: {repo_id}/{filename}")
        config_path = hf_hub_download(repo_id=repo_id, filename=f"{filename}")
        from_huggingface = True
    elif "/" in config_file and not os.path.exists(config_file) and not config_file.endswith(".json"):
        # If format is "username/repo-name", default to "config.json"
        print(f"Loading config from Hugging Face Hub: {config_file}/configs/config.json")
        config_path = hf_hub_download(repo_id=config_file, filename="configs/config.json")
        from_huggingface = True
    else:
        # Local file path
        config_path = config_file
        from_huggingface = False

    with open(config_path) as f:
        _config = json.load(f)

    if from_huggingface:
        _config["model_load_path"] = config_file
        _config['statistics_path'] = config_file

    config = {}
    if _config.get("parent"):
        deep_update(config, load_config(_config["parent"]))
    deep_update(config, _config)
    _resolve_config_paths(config, REPO_ROOT)
    return config