"""Write the result table as XLSX or CSV, plus a separate errors file for failed rows."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from openpyxl import Workbook

from dxa_qc.report.schema import REPORT_COLUMNS, ProcessingStatus, ResultRow

# Spreadsheet apps execute cells starting with these as formulas (CSV/formula injection).
# Paths come from user-supplied archives, so they are untrusted.
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


class ReportFormat(StrEnum):
    XLSX = "xlsx"
    CSV = "csv"


@dataclass(frozen=True, slots=True)
class WrittenReport:
    table: Path
    errors: Path | None


def neutralize_formula(value: object) -> object:
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def _rows(results: Sequence[ResultRow]) -> list[list[object]]:
    return [[neutralize_formula(r.table_values()[c]) for c in REPORT_COLUMNS] for r in results]


def _write_xlsx(path: Path, header: Sequence[str], rows: list[list[object]]) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "results"
    ws.append(list(header))
    for row in rows:
        ws.append(row)
    wb.save(path)


def _write_csv(path: Path, header: Sequence[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows([["" if v is None else v for v in row] for row in rows])


def write_report(
    results: Sequence[ResultRow], output_dir: Path, fmt: ReportFormat, stem: str = "results"
) -> WrittenReport:
    """Write ``<stem>.<fmt>`` and, if anything failed, ``<stem>_errors.csv``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = output_dir / f"{stem}.{fmt.value}"
    rows = _rows(results)
    if fmt is ReportFormat.XLSX:
        _write_xlsx(table, REPORT_COLUMNS, rows)
    else:
        _write_csv(table, REPORT_COLUMNS, rows)

    failed = [r for r in results if r.processing_status is ProcessingStatus.FAILURE]
    errors_path = None
    if failed:
        errors_path = output_dir / f"{stem}_errors.csv"
        _write_csv(
            errors_path,
            ("path_to_study", "error"),
            [[neutralize_formula(r.path_to_study), neutralize_formula(r.error or "")] for r in failed],
        )
    return WrittenReport(table=table, errors=errors_path)
