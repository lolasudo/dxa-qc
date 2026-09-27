"""Output row schema: the organizer's table plus ``quality_prob``."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dxa_qc.domain.taxonomy import AnatomicalRegion


class ProcessingStatus(StrEnum):
    SUCCESS = "Success"
    FAILURE = "Failure"


# Column order is part of the contract with the organizer and the frontend.
REPORT_COLUMNS: tuple[str, ...] = (
    "path_to_study",
    "study_uid",
    "image_uid",
    "anatomical_region",
    "quality_class",
    "quality_prob",
    "violation_type",
    "processing_status",
    "time_of_processing",
)


class ResultRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path_to_study: str
    study_uid: str = ""
    image_uid: str = ""
    anatomical_region: AnatomicalRegion | None = None
    quality_class: int | None = Field(default=None, ge=0, le=1)
    quality_prob: float | None = Field(default=None, ge=0.0, le=1.0)
    violation_type: str = ""
    processing_status: ProcessingStatus
    time_of_processing: float = Field(ge=0.0)
    # Not part of the table: goes to the separate errors file.
    error: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ResultRow:
        if self.processing_status is ProcessingStatus.SUCCESS:
            if self.anatomical_region is None or self.quality_class is None or self.quality_prob is None:
                raise ValueError("successful row needs anatomical_region, quality_class and quality_prob")
            if self.quality_class == 0 and self.violation_type:
                raise ValueError("quality_class=0 contradicts a non-empty violation_type")
            if self.error:
                raise ValueError("successful row must not carry an error")
        elif not self.error:
            raise ValueError("failed row must explain the failure in `error`")
        return self

    def table_values(self) -> dict[str, object]:
        return {
            "path_to_study": self.path_to_study,
            "study_uid": self.study_uid,
            "image_uid": self.image_uid,
            "anatomical_region": self.anatomical_region.value if self.anatomical_region else "",
            "quality_class": self.quality_class,
            "quality_prob": None if self.quality_prob is None else round(self.quality_prob, 6),
            "violation_type": self.violation_type,
            "processing_status": self.processing_status.value,
            "time_of_processing": round(self.time_of_processing, 3),
        }
