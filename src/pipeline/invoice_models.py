"""Pydantic mirror for the canonical invoice_extraction.v1 JSON contract."""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


CurrencyCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
DateString = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$")]
TimeString = Annotated[str, StringConstraints(pattern=r"^\d{2}:\d{2}(:\d{2})?$")]
DateTimeString = Annotated[
    str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _assert_unique(values: list[str], field_path: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_path} must contain unique values")


class Source(StrictModel):
    sample_id: Annotated[str, StringConstraints(min_length=1)]
    img_id: str | None
    image_path: str | None
    raw_text: str


class Party(StrictModel):
    name: str | None
    address: str | None
    tax_code: str | None
    phone: str | None


class InvoiceMetadata(StrictModel):
    invoice_number: str | None
    invoice_series: str | None
    issued_at: DateTimeString | None
    issued_date: DateString | None
    issued_time: TimeString | None
    currency: CurrencyCode | None
    payment_method: str | None

    @model_validator(mode="after")
    def validate_temporal_consistency(self) -> "InvoiceMetadata":
        parsed_date = None
        parsed_time = None
        parsed_datetime = None

        if self.issued_date is not None:
            parsed_date = date.fromisoformat(self.issued_date)
        if self.issued_time is not None:
            parsed_time = time.fromisoformat(self.issued_time)
        if self.issued_at is not None:
            parsed_datetime = datetime.fromisoformat(self.issued_at)

        if self.issued_date is not None and self.issued_time is not None and self.issued_at is None:
            raise ValueError("invoice.issued_at is required when issued_date and issued_time exist")
        if self.issued_at is not None and (self.issued_date is None or self.issued_time is None):
            raise ValueError("invoice.issued_date and issued_time are required when issued_at exists")
        if parsed_datetime is not None and parsed_date is not None and parsed_datetime.date() != parsed_date:
            raise ValueError("invoice.issued_at date component conflicts with issued_date")
        if parsed_datetime is not None and parsed_time is not None and parsed_datetime.time() != parsed_time:
            raise ValueError("invoice.issued_at time component conflicts with issued_time")
        return self


class Money(StrictModel):
    raw: str | None
    value: float | None = Field(default=None, ge=0)
    currency: CurrencyCode | None


class Quantity(StrictModel):
    raw: str | None
    value: float | None = Field(default=None, ge=0)
    unit: str | None


class Totals(StrictModel):
    subtotal: Money
    discount: Money
    vat_amount: Money
    total_amount: Money


class TaxLine(StrictModel):
    tax_name: str | None
    tax_rate: float | None = Field(default=None, ge=0, le=100)
    taxable_amount: Money
    tax_amount: Money


class LineItem(StrictModel):
    line_number: int = Field(ge=1)
    description: str | None
    sku: str | None
    quantity: Quantity
    unit_price: Money
    line_total: Money
    vat_rate: float | None = Field(default=None, ge=0, le=100)


class FieldEvidence(StrictModel):
    field_path: Annotated[str, StringConstraints(min_length=1)]
    raw_text: Annotated[str, StringConstraints(min_length=1)]
    source_label: str | None


class ExtractionMetadata(StrictModel):
    status: Literal["ok", "partial", "failed"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    missing_fields: list[Annotated[str, StringConstraints(min_length=1)]]
    warnings: list[str]
    field_evidence: list[FieldEvidence]

    @model_validator(mode="after")
    def validate_status_and_uniqueness(self) -> "ExtractionMetadata":
        _assert_unique(self.missing_fields, "extraction.missing_fields")
        _assert_unique(self.warnings, "extraction.warnings")
        if self.status == "ok" and self.missing_fields:
            raise ValueError("extraction.status=ok requires empty missing_fields")
        if self.status == "partial" and not self.missing_fields:
            raise ValueError("extraction.status=partial requires at least one missing field")
        return self


class InvoiceExtraction(StrictModel):
    schema_version: Literal["invoice_extraction.v1"]
    document_type: Literal["retail_receipt", "vat_invoice", "unknown"]
    source: Source
    seller: Party
    buyer: Party
    invoice: InvoiceMetadata
    totals: Totals
    taxes: list[TaxLine]
    items: list[LineItem]
    extraction: ExtractionMetadata

    @model_validator(mode="after")
    def validate_cross_object_rules(self) -> "InvoiceExtraction":
        line_numbers = [item.line_number for item in self.items]
        if len(line_numbers) != len(set(line_numbers)):
            raise ValueError("items.line_number values must be unique")

        for index, evidence in enumerate(self.extraction.field_evidence):
            if evidence.raw_text not in self.source.raw_text:
                raise ValueError(
                    f"extraction.field_evidence[{index}].raw_text is not a source substring"
                )
        return self


def validate_invoice_payload(payload: object) -> InvoiceExtraction:
    """Validate a raw object against the typed invoice_extraction.v1 contract."""
    return InvoiceExtraction.model_validate(payload)
