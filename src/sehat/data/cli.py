"""``sehat-data`` command line interface.

Everything here runs on the standard library alone, so the whole data layer —
downloading public datasets, building the unified manifest with leakage-safe
splits, verifying integrity — works on a bare Python install in a clinic with
no package manager access.

Usage::

    sehat-data list
    sehat-data download --dataset shenzhen [--data-dir data]
    sehat-data build-manifest [--datasets shenzhen montgomery] \\
        [--strategy patient | site-holdout] [--out data/manifest.csv]
    sehat-data verify --manifest data/manifest.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from sehat.data import config as data_config
from sehat.data.datasets import get_dataset, list_datasets
from sehat.data.manifest import (
    ManifestValidationError,
    validate_manifest_csv,
    write_manifest_csv,
)
from sehat.data.splits import (
    DEFAULT_SEED,
    DEFAULT_TEST_FRAC,
    DEFAULT_VAL_FRAC,
    holdout_assignment,
    patient_strata,
    stratified_group_assignment,
)
from sehat.data.versioning import dvc_add, verify_sidecar, write_sidecar

__all__ = ["main"]

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sehat-data",
        description="Download, normalize, split, and version Sehat's public datasets.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list registered datasets")

    download = sub.add_parser("download", help="download + extract one dataset")
    download.add_argument("--dataset", required=True, choices=list_datasets())
    download.add_argument("--data-dir", type=Path, default=Path("data"))
    download.add_argument("--config", type=Path, default=None, help="datasets.yaml override")

    build = sub.add_parser("build-manifest", help="build the unified manifest CSV")
    build.add_argument(
        "--datasets",
        nargs="+",
        choices=list_datasets(),
        default=["shenzhen", "montgomery"],
    )
    build.add_argument("--data-dir", type=Path, default=Path("data"))
    build.add_argument("--out", type=Path, default=Path("data/manifest.csv"))
    build.add_argument("--config", type=Path, default=None, help="datasets.yaml override")
    build.add_argument("--strategy", choices=["patient", "site-holdout"], default=None)
    build.add_argument("--holdout-site", default=None, help="site to reserve as test_external")
    build.add_argument("--val-frac", type=float, default=None)
    build.add_argument("--test-frac", type=float, default=None)
    build.add_argument("--seed", type=int, default=None)

    verify = sub.add_parser("verify", help="validate a manifest CSV and its checksum")
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def _cmd_list() -> int:
    for name in list_datasets():
        dataset = get_dataset(name)
        print(
            f"{name:12s} site={dataset.spec.site:10s} disease={dataset.spec.disease:10s} "
            f"{dataset.spec.homepage}"
        )
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    overrides = data_config.dataset_overrides(args.dataset, args.config)
    dataset = get_dataset(args.dataset, overrides)
    dataset.download(args.data_dir / "raw" / args.dataset)
    extracted = dataset.extract(args.data_dir / "raw" / args.dataset)
    print(f"{args.dataset}: extracted at {extracted}")
    return 0


def _cmd_build_manifest(args: argparse.Namespace) -> int:
    defaults = data_config.split_defaults()

    def pick(cli_value: object, key: str, fallback: object) -> object:
        """Precedence: CLI flag > splits.yaml > built-in default."""
        return cli_value if cli_value is not None else defaults.get(key, fallback)

    strategy = pick(args.strategy, "strategy", "patient")
    holdout_site = pick(args.holdout_site, "holdout_site", "montgomery")
    val_frac = float(pick(args.val_frac, "val_frac", DEFAULT_VAL_FRAC))
    test_frac = float(pick(args.test_frac, "test_frac", DEFAULT_TEST_FRAC))
    seed = int(pick(args.seed, "seed", DEFAULT_SEED))

    rows: list[dict[str, str]] = []
    for name in args.datasets:
        dataset = get_dataset(name, data_config.dataset_overrides(name, args.config))
        raw_dir = args.data_dir / "raw" / name
        dataset.download(raw_dir)
        dataset_rows = list(dataset.iter_records(dataset.extract(raw_dir)))
        logger.info("%s: %d images", name, len(dataset_rows))
        rows.extend(dataset_rows)
    if not rows:
        print("no records found; nothing to write", file=sys.stderr)
        return 1

    if strategy == "site-holdout":
        assignment = holdout_assignment(
            ((r["patient_id"], r["site"], int(r["label"])) for r in rows),
            holdout_site,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
        )
    else:
        strata = patient_strata((r["patient_id"], int(r["label"])) for r in rows)
        assignment = stratified_group_assignment(
            strata, val_frac=val_frac, test_frac=test_frac, seed=seed
        )
    for row in rows:
        row["split"] = assignment[row["patient_id"]]

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1

    try:
        write_manifest_csv(rows, args.out)
    except ManifestValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    sidecar = write_sidecar(args.out)
    dvc_add([args.data_dir])
    print(f"wrote {len(rows)} rows to {args.out} (sha256 sidecar: {sidecar})")
    print("splits: " + ", ".join(f"{split}={count}" for split, count in sorted(counts.items())))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    try:
        validate_manifest_csv(args.manifest)
    except ManifestValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    checksum = "ok" if verify_sidecar(args.manifest) else "no sidecar (or mismatch)"
    print(f"{args.manifest}: schema valid; checksum: {checksum}")
    return 0 if checksum == "ok" else 2


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``sehat-data`` console script."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.command == "list":
        return _cmd_list()
    if args.command == "download":
        return _cmd_download(args)
    if args.command == "build-manifest":
        return _cmd_build_manifest(args)
    if args.command == "verify":
        return _cmd_verify(args)
    return 1  # pragma: no cover - argparse enforces a valid command


if __name__ == "__main__":
    sys.exit(main())
