"""Montgomery County, Maryland TB chest X-ray set (NLM).

~138 frontal chest X-rays (58 abnormal with TB manifestations, 80 normal)
collected by the Montgomery County Department of Health and published by the
U.S. National Library of Medicine. Images are named ``MCUCXR_XXXX_F.png`` with
``F=1`` indicating TB. Public domain; cite Jaeger et al., 2014.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from sehat.data.datasets._nlm import iter_nlm_tb_records
from sehat.data.datasets.base import DatasetSpec, ManifestRecord, RawDataset

__all__ = ["MontgomeryDataset"]

_FILENAME_RE = re.compile(r"^(MCUCXR_\d{4})_([01])\.png$", re.IGNORECASE)


class MontgomeryDataset(RawDataset):
    """NLM Montgomery County TB collection."""

    spec = DatasetSpec(
        name="montgomery",
        site="montgomery",
        disease="tb",
        urls=("https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip",),
        sha256=(None,),  # NLM does not publish checksums; record after first verified download
        homepage="https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/tuberculosis-chest-x-ray-image-data-sets.html",
        license="Public domain (NIH/NLM); attribution required",
        citation=(
            "Jaeger S, Candemir S, Antani S, Wang YX, Lu PX, Thoma G. Two public chest "
            "X-ray datasets for computer-aided screening of pulmonary diseases. "
            "Quant Imaging Med Surg. 2014;4(6):475-7."
        ),
    )

    def iter_records(self, extracted_dir: str | Path) -> Iterator[ManifestRecord]:
        """Yield one record per chest X-ray, skipping mask images."""
        yield from iter_nlm_tb_records(extracted_dir, site=self.spec.site, filename_re=_FILENAME_RE)
