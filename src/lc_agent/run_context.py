from dataclasses import dataclass
from datetime import date
from typing import Optional

@dataclass(frozen=True)
class RunContext:
    today: str
    current_year: int


def make_run_context(today: Optional[date] = None) -> RunContext:
    d = today or date.today()
    y = d.year
    return RunContext(
        today=d.isoformat(),
        current_year=y
    )
