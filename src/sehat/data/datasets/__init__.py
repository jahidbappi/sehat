"""Public dataset plugins and registry.

Each plugin fetches one public dataset (resumable, checksummed), extracts it,
and normalizes it into unified manifest records. Raw data always lands under
the gitignored ``data/`` directory, never inside the package.
"""

from __future__ import annotations

from collections.abc import Mapping

from sehat.data.datasets.base import DatasetSpec, ManifestRecord, RawDataset
from sehat.data.datasets.montgomery import MontgomeryDataset
from sehat.data.datasets.nih_cxr import NIHChestXrayDataset
from sehat.data.datasets.shenzhen import ShenzhenDataset

__all__ = [
    "DATASETS",
    "DatasetSpec",
    "ManifestRecord",
    "MontgomeryDataset",
    "NIHChestXrayDataset",
    "RawDataset",
    "ShenzhenDataset",
    "get_dataset",
    "list_datasets",
]

DATASETS: dict[str, type[RawDataset]] = {
    "shenzhen": ShenzhenDataset,
    "montgomery": MontgomeryDataset,
    "nih_cxr": NIHChestXrayDataset,
}


def list_datasets() -> list[str]:
    """Return the names of all registered datasets."""
    return sorted(DATASETS)


def get_dataset(name: str, overrides: Mapping[str, object] | None = None) -> RawDataset:
    """Instantiate a registered dataset plugin, optionally with config overrides."""
    try:
        dataset_cls = DATASETS[name]
    except KeyError:
        raise ValueError(
            f"unknown dataset {name!r}; available: {', '.join(list_datasets())}"
        ) from None
    return dataset_cls(overrides)
