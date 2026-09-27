"""Russian explanation of a result: what was found and the measured facts.

Numbers come from the measurements, never invented. Wording is cautious on purpose: the
service flags studies for a specialist, it does not replace their judgement.
"""

from __future__ import annotations

import re

from dxa_qc.domain.taxonomy import ALLOWED_VIOLATIONS, AnatomicalRegion, HipSide, Violation
from dxa_qc.pipeline.assessment import QualityAssessment

# Hints phrased as what to check on the image.
_HINTS: dict[tuple[AnatomicalRegion, Violation], str] = {
    (AnatomicalRegion.SPINE, Violation.POSITIONING): (
        "проверьте, что в кадре видны верхние края подвздошных костей и половина тела Th12"
    ),
    (AnatomicalRegion.SPINE, Violation.SPINE_AXIS): "ось позвоночника должна быть выровнена по вертикали",
    (AnatomicalRegion.SPINE, Violation.FOREIGN_OBJECTS): (
        "в зоне сканирования обнаружены яркие инородные структуры (например, металл одежды)"
    ),
    (AnatomicalRegion.HIP, Violation.POSITIONING): (
        "проверьте ротацию бедра по малому вертелу и видимость большого вертела, шейки и седалищной кости"
    ),
    (AnatomicalRegion.HIP, Violation.HIP_ROI): (
        "поле должно захватывать не менее 3 см выше и ниже области интереса и 2 см сбоку"
    ),
}

_FAILURES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"not a valid DICOM|no DICOM header|DICM", re.I), "Файл не является корректным DICOM."),
    (re.compile(r"region not recognized", re.I), "Не удалось уверенно определить анатомическую область."),
    (re.compile(r"too large", re.I), "Файл превышает допустимый размер."),
    (
        re.compile(r"PixelData|decode pixel|pixel data", re.I),
        "Не удалось прочитать изображение в DICOM-файле.",
    ),
    (
        re.compile(r"unsupported|expected", re.I),
        "Формат изображения не поддерживается (ожидается одно полутоновое изображение).",
    ),
)


def _pct(p: float) -> str:
    return f"{round(100 * p)}%"


def _num(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def explain_failure(error: str | None) -> str:
    for pattern, text in _FAILURES:
        if error and pattern.search(error):
            return text
    return "Ошибка обработки файла. Подробности - в журнале ошибок пакета."


_SIDES = {HipSide.RIGHT: "правое бедро", HipSide.LEFT: "левое бедро"}


def describe_projection(region: AnatomicalRegion, side: HipSide | None) -> str:
    """Both regions are acquired in the frontal (AP) projection, the only one in the training
    data; the hip side comes from the region classifier."""
    if region is AnatomicalRegion.HIP and side is not None:
        return f"Прямая (AP), {_SIDES[side]}"
    return "Прямая (AP)"


def explain(region: AnatomicalRegion, quality: QualityAssessment, max_axis_tilt_deg: float) -> str:
    parts: list[str] = []
    ordered = [v for v in ALLOWED_VIOLATIONS[region] if v in quality.violations]
    if ordered:
        found = "; ".join(
            f"{v.value.lower()} ({_pct(quality.criterion_probs[v])})"
            if v in quality.criterion_probs
            else v.value.lower()
            for v in ordered
        )
        parts.append(f"Выявлено: {found}.")
        parts.extend(f"{hint[0].upper()}{hint[1:]}." for v in ordered if (hint := _HINTS.get((region, v))))
    elif quality.quality_class == 1:
        parts.append(
            "Общая оценка указывает на нарушение качества, но ни один отдельный критерий не превысил "
            "порог - рекомендуется визуальная проверка специалистом."
        )
    else:
        parts.append("Нарушений качества не выявлено.")

    f = quality.findings
    if region is AnatomicalRegion.SPINE and "axis_tilt_deg" in f:
        tilt = abs(float(f["axis_tilt_deg"]))
        parts.append(f"Наклон оси позвоночника: {_num(tilt)}° (допустимо до {_num(max_axis_tilt_deg, 0)}°).")
    if region is AnatomicalRegion.HIP:
        if f.get("prosthesis_suspected"):
            parts.append(
                "Обнаружен металл, характерный для эндопротеза: стандартные критерии укладки "
                "к такому снимку применимы ограниченно."
            )
        if "field_height_mm" in f and "field_width_mm" in f:
            parts.append(
                f"Размер поля сканирования: {_num(float(f['field_width_mm']) / 10)} x "
                f"{_num(float(f['field_height_mm']) / 10)} см."
            )
    parts.append(f"Вероятность нарушения качества: {_pct(quality.quality_prob)}.")
    return " ".join(parts)
