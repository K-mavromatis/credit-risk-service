"""Domain model for one accepted loan and the raw-row -> model conversion.

This is the only place that knows what a *valid* record looks like. Every
other stage of the streaming pipeline just asks `to_accepted_loan()` and drops
whatever comes back as ``None``.

Fields follow the LendingClub "accepted loans" columns used by the scoring
model (see ``notebooks/01_data_preparation.ipynb``); ``loan_id`` is the
original ``id`` column.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class AcceptedLoan(BaseModel):
    """One validated loan record.

    Raw values may arrive as CSV-style strings (``" 36 months"``, ``"3600.0"``);
    Pydantic coerces them, and anything that cannot be coerced is rejected.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    loan_id: str = Field(min_length=1)
    loan_amnt: float = Field(gt=0)
    term: int = Field(gt=0, description="Loan term in months.")
    int_rate: float = Field(ge=0)
    grade: str = Field(pattern=r"^[A-G]$")
    addr_state: str = Field(min_length=2, max_length=2)
    loan_status: str = Field(min_length=1)
    issue_d: str = Field(min_length=1)
    annual_inc: float | None = Field(default=None, ge=0)

    @field_validator("term", mode="before")
    @classmethod
    def _term_in_months(cls, value: Any) -> Any:
        """Turn ``" 36 months"`` into ``"36"``; other values pass through."""
        if isinstance(value, str):
            return value.strip().split(" ", 1)[0]
        return value

    @field_validator("annual_inc", mode="before")
    @classmethod
    def _blank_income_is_missing(cls, value: Any) -> Any:
        """An empty cell means "income unknown", not "invalid record"."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def key(self) -> tuple[str, str]:
        """Compact deduplication key: the same loan id in the same issue month.

        Built from the already-normalised (stripped) fields, so
        ``" 123"`` and ``"123"`` collide as they should.
        """
        return (self.loan_id, self.issue_d)


def to_accepted_loan(row: object) -> AcceptedLoan | None:
    """Validate one raw row; return ``None`` for anything malformed.

    ``row`` is deliberately typed ``object``: a JSON line may decode to a list,
    a number or even a plain string, and all of those must be rejected, not
    crash the pipeline.
    """
    try:
        return AcceptedLoan.model_validate(row)
    except ValidationError:
        return None
