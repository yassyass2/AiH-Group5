"""Preprocessing package for the APTOS 2019 grading pipeline."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DEFAULT_IMAGE_SIZE": "dr_grading.preprocessing.transforms",
    "apply_clahe": "dr_grading.preprocessing.transforms",
    "apply_oversampling": "dr_grading.preprocessing.sampling",
    "build_dataset": "dr_grading.preprocessing.pipeline",
    "create_retina_mask": "dr_grading.preprocessing.transforms",
    "crop_black_border": "dr_grading.preprocessing.transforms",
    "extract_green_channel_clahe": "dr_grading.preprocessing.transforms",
    "is_cut_off": "dr_grading.preprocessing.transforms",
    "load_split_df": "dr_grading.preprocessing.runner",
    "main": "dr_grading.preprocessing.runner",
    "parse_args": "dr_grading.preprocessing.runner",
    "preprocess_image": "dr_grading.preprocessing.transforms",
    "preprocess_image_path": "dr_grading.preprocessing.transforms",
    "reduce_fundus_noise": "dr_grading.preprocessing.transforms",
    "run_pipeline": "dr_grading.preprocessing.pipeline",
    "run_preprocessing": "dr_grading.preprocessing.runner",
    "write_preprocessing_manifest": "dr_grading.preprocessing.runner",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
