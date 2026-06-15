"""Backward-compatible entry point for preprocessing orchestration."""

from dr_grading.preprocessing.runner import (
    load_split_df,
    main,
    parse_args,
    run_preprocessing,
    write_preprocessing_manifest,
)

__all__ = [
    "load_split_df",
    "main",
    "parse_args",
    "run_preprocessing",
    "write_preprocessing_manifest",
]
