"""Shared parsing for the NLM tuberculosis chest X-ray sets (Shenzhen, Montgomery).

Both datasets ship PNGs named ``<PATIENT>_<FLAG>.png`` where ``FLAG`` is ``0``
(normal) or ``1`` (TB), alongside per-image clinical reading text files whose
first lines contain patient sex and age. The layouts are near-identical, so the
record-walking logic lives here; each dataset module only supplies its spec and
filename pattern.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path

from sehat.data.datasets.base import ManifestRecord

__all__ = ["iter_nlm_tb_records", "parse_clinical_reading"]

logger = logging.getLogger(__name__)

_SEX_RE = re.compile(r"(?:sex|gender)\s*:?\s*([MF])\b", re.IGNORECASE)
_AGE_RE = re.compile(r"age\s*:?\s*(\d{1,3})", re.IGNORECASE)


def parse_clinical_reading(text: str) -> dict[str, str]:
    """Extract ``sex`` and ``age`` from a clinical reading file, leniently."""
    sex_match = _SEX_RE.search(text)
    age_match = _AGE_RE.search(text)
    age = str(int(age_match.group(1))) if age_match else ""
    if age and not 0 <= int(age) <= 120:
        age = ""
    return {
        "sex": sex_match.group(1).upper() if sex_match else "unknown",
        "age": age,
    }


def iter_nlm_tb_records(
    extracted_dir: str | Path,
    *,
    site: str,
    filename_re: re.Pattern[str],
) -> Iterator[ManifestRecord]:
    """Yield manifest records for an NLM TB set.

    ``filename_re`` must capture group 1 = patient id, group 2 = label flag
    (``"0"`` normal, ``"1"`` TB) from the PNG filename. Files that do not match
    (masks, readme files) are skipped.
    """
    extracted_dir = Path(extracted_dir)
    readings = {path.stem: path for path in extracted_dir.rglob("*.txt")}
    for png in sorted(extracted_dir.rglob("*.png")):
        match = filename_re.match(png.name)
        if not match:
            logger.debug("skipping non-CXR file %s", png)
            continue
        patient_id, flag = match.group(1), match.group(2)
        meta = {"sex": "unknown", "age": ""}
        if png.stem in readings:
            meta = parse_clinical_reading(readings[png.stem].read_text(errors="replace"))
        yield {
            "image_path": str(png.relative_to(extracted_dir)),
            "label": flag,
            "disease": "tb" if flag == "1" else "normal",
            "site": site,
            "patient_id": patient_id,
            "split": "",
            "sex": meta["sex"],
            "age": meta["age"],
        }
