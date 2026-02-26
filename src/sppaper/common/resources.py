from importlib.resources import files as importlib_resources_files
from pathlib import Path


def get_inputs_dir() -> Path:
    return Path(str(importlib_resources_files("sppaper").parent.parent / "input_data"))


def get_outputs_dir() -> Path:
    path = Path(str(importlib_resources_files("sppaper").parent.parent / "output_data"))
    path.mkdir(exist_ok=True, parents=True)
    return path


def get_poseforge_datadir() -> Path:
    return Path(str(importlib_resources_files("poseforge").parent.parent / "bulk_data"))


def get_flygym_assetdir() -> Path:
    from flygym import assets_dir

    return assets_dir


def get_spotlight_trials_dir() -> Path:
    return get_inputs_dir() / "spotlight_trials"
