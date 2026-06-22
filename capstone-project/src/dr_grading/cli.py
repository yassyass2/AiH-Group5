from __future__ import annotations

import argparse
from pathlib import Path

from dr_grading.paths import DEFAULT_TRAINING_CONFIG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dr-grading",
        description="APTOS 2019 diabetic retinopathy grading research pipeline.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    preprocess = subcommands.add_parser(
        "preprocess",
        help="Build model-ready arrays from the APTOS 2019 split files.",
    )
    preprocess.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for preprocessed .npy arrays.",
    )
    preprocess.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Existing APTOS dataset directory; defaults to kagglehub download.",
    )
    preprocess.set_defaults(handler=_run_preprocess)

    train = subcommands.add_parser(
        "train",
        help="Train and evaluate the EfficientNet-B0 DR classifier.",
    )
    train.add_argument("--config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    train.set_defaults(handler=_run_train)

    predict = subcommands.add_parser(
        "predict",
        help="Run a saved .keras model on one preprocessed image sample.",
    )
    predict.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model filename, full path, or 1-based model number from the discovered list.",
    )
    predict.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Preprocessed dataset source: a .npz archive or directory of .npy arrays.",
    )
    predict.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory that contains saved .keras models.",
    )
    predict.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split to sample from.",
    )
    predict.add_argument(
        "--index",
        type=int,
        default=None,
        help="0-based sample index within the selected split.",
    )
    predict.add_argument(
        "--random",
        action="store_true",
        help="Choose a random sample from the selected split.",
    )
    predict.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when selecting a random sample.",
    )
    predict.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many class probabilities to print.",
    )
    predict.add_argument(
        "--list-models",
        action="store_true",
        help="List discovered .keras models and exit.",
    )
    predict.set_defaults(handler=_run_predict)

    return parser


def _run_preprocess(args: argparse.Namespace) -> None:
    from dr_grading.preprocess import run_preprocessing

    run_preprocessing(output_dir=args.output_dir, dataset_dir=args.dataset_dir)


def _run_train(args: argparse.Namespace) -> None:
    from dr_grading.train import main as train_main

    train_main(["--config", str(args.config)])


def _run_predict(args: argparse.Namespace) -> None:
    from dr_grading.predict import main as predict_main

    forwarded_args: list[str] = []
    if args.model is not None:
        forwarded_args.extend(["--model", args.model])
    if args.data is not None:
        forwarded_args.extend(["--data", str(args.data)])
    if args.models_dir is not None:
        forwarded_args.extend(["--models-dir", str(args.models_dir)])
    if args.split is not None:
        forwarded_args.extend(["--split", args.split])
    if args.index is not None:
        forwarded_args.extend(["--index", str(args.index)])
    if args.random:
        forwarded_args.append("--random")
    if args.seed is not None:
        forwarded_args.extend(["--seed", str(args.seed)])
    if args.top_k is not None:
        forwarded_args.extend(["--top-k", str(args.top_k)])
    if args.list_models:
        forwarded_args.append("--list-models")
    predict_main(forwarded_args)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
