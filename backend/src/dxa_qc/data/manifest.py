"""Training manifest: one CSV row per image - the format for datasets of any size and origin.

Columns (UTF-8, header required)::

    path,study_id,region,side,overall,positioning,spine_axis,foreign_objects,hip_roi,comment

* ``path`` - DICOM file relative to the images root (absolute paths and ``..`` are rejected);
* ``study_id`` - groups images of one study/patient: they never land in different CV folds;
* ``region`` - ``spine`` | ``hip`` | empty (then detected by the region classifier);
* ``side`` - ``left`` | ``right`` for hips (empty -> detected), empty for spine;
* ``overall`` - expert verdict 1 = has a quality problem, 0 = good, empty = unlabeled image;
* criterion columns - 1 = violated, 0 = fine; required when ``overall`` is set, and must stay
  empty for criteria that do not apply to the region (``spine_axis``/``foreign_objects`` for hips,
  ``hip_roi`` for spine).

``python -m dxa_qc.training.make_manifest`` converts the organizer's workbook into this format.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from dxa_qc.domain.taxonomy import ALLOWED_VIOLATIONS, AnatomicalRegion, HipSide, Violation

COLUMNS = (
    "path",
    "study_id",
    "region",
    "side",
    "overall",
    "positioning",
    "spine_axis",
    "foreign_objects",
    "hip_roi",
    "comment",
)
CRITERION_COLUMNS: dict[Violation, str] = {
    Violation.POSITIONING: "positioning",
    Violation.SPINE_AXIS: "spine_axis",
    Violation.FOREIGN_OBJECTS: "foreign_objects",
    Violation.HIP_ROI: "hip_roi",
}
_REGIONS = {"spine": AnatomicalRegion.SPINE, "hip": AnatomicalRegion.HIP}
_SIDES = {"left": HipSide.LEFT, "right": HipSide.RIGHT}


class ManifestError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ManifestRow:
    line: int
    path: Path  # resolved, guaranteed inside the images root
    study_id: str
    region: AnatomicalRegion | None
    side: HipSide | None
    overall: int | None
    violations: frozenset[Violation]
    comment: str = ""

    @property
    def labeled(self) -> bool:
        return self.overall is not None


def _flag(value: str, where: str) -> int | None:
    value = value.strip()
    if value == "":
        return None
    if value not in ("0", "1"):
        raise ManifestError(f"{where}: expected 0, 1 or empty, got {value!r}")
    return int(value)


def _resolve(root: Path, raw: str, where: str) -> Path:
    rel = PurePosixPath(raw.strip().replace("\\", "/"))
    if not raw.strip() or rel.is_absolute() or ".." in rel.parts or (rel.parts and ":" in rel.parts[0]):
        raise ManifestError(f"{where}: path must be relative to the images root without '..': {raw!r}")
    path = (root / Path(*rel.parts)).resolve()
    if not path.is_relative_to(root):
        raise ManifestError(f"{where}: path escapes the images root: {raw!r}")
    return path


def _parse_row(row: dict[str, str], line: int, root: Path) -> ManifestRow:
    where = f"line {line}"
    region_raw = row["region"].strip().lower()
    side_raw = row["side"].strip().lower()
    if region_raw and region_raw not in _REGIONS:
        raise ManifestError(f"{where}: region must be spine, hip or empty, got {region_raw!r}")
    if side_raw and side_raw not in _SIDES:
        raise ManifestError(f"{where}: side must be left, right or empty, got {side_raw!r}")
    region = _REGIONS.get(region_raw)
    if region is AnatomicalRegion.SPINE and side_raw:
        raise ManifestError(f"{where}: spine rows must not have a side")
    study = row["study_id"].strip()
    if not study:
        raise ManifestError(f"{where}: study_id is required (it keeps a study inside one CV fold)")

    overall = _flag(row["overall"], f"{where} overall")
    flags = {v: _flag(row[col], f"{where} {col}") for v, col in CRITERION_COLUMNS.items()}
    if overall is None:
        if any(f is not None for f in flags.values()):
            raise ManifestError(f"{where}: criteria filled but overall is empty")
    else:
        if region is None:
            raise ManifestError(f"{where}: labeled rows need an explicit region (criteria depend on it)")
        allowed = ALLOWED_VIOLATIONS[region]
        for v, f in flags.items():
            if v in allowed and f is None:
                raise ManifestError(f"{where}: {CRITERION_COLUMNS[v]} is required for {region.name.lower()}")
            if v not in allowed and f is not None:
                raise ManifestError(
                    f"{where}: {CRITERION_COLUMNS[v]} does not apply to {region.name.lower()}"
                )
    return ManifestRow(
        line=line,
        path=_resolve(root, row["path"], where),
        study_id=study,
        region=region,
        side=_SIDES.get(side_raw),
        overall=overall,
        violations=frozenset(v for v, f in flags.items() if f == 1),
        comment=(row.get("comment") or "").strip(),
    )


def load_manifest(path: Path, images_root: Path | None = None) -> list[ManifestRow]:
    """Parse and validate the whole manifest; errors carry the line number.

    ``images_root`` defaults to the manifest's directory.
    """
    path = Path(path)
    root = Path(images_root or path.parent).resolve()
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ManifestError(f"manifest {path} lacks columns: {', '.join(missing)}")
        rows = [_parse_row(r, n, root) for n, r in enumerate(reader, start=2)]
    if not rows:
        raise ManifestError(f"manifest {path} has no rows")
    seen: dict[Path, int] = {}
    for r in rows:
        if r.path in seen:
            raise ManifestError(f"line {r.line}: duplicate path (first at line {seen[r.path]})")
        seen[r.path] = r.line
    return rows


def write_manifest(rows: Iterable[dict[str, str]], path: Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in COLUMNS})
