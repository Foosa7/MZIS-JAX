"""Request and response shapes for the control API.

`user_id` is deliberately absent from every request body. It used to be
client-supplied, which meant a caller could submit work as anyone; it now
comes from the authenticated Tailscale identity and is never read from the
payload.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

MAX_TARGETS_PER_JOB = 64
MAX_MODES = 32


class HeaterCurrents(BaseModel):
    heater_theta: float
    heater_phi: float


class UnitaryTarget(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    matrix_real: List[List[float]]
    matrix_imag: List[List[float]]
    initial_currents_ma: Dict[str, HeaterCurrents] = Field(default_factory=dict)

    @field_validator("matrix_real", "matrix_imag")
    @classmethod
    def _square_and_bounded(cls, value):
        if not value:
            raise ValueError("matrix must not be empty")
        n = len(value)
        if n > MAX_MODES:
            raise ValueError(f"matrix is {n}x{n}, larger than the {MAX_MODES} mode limit")
        if any(len(row) != n for row in value):
            raise ValueError("matrix must be square")
        return value


class JobRequest(BaseModel):
    """What a client submits. The server assigns identity and job id."""

    job_id: Optional[str] = Field(default=None, max_length=64)
    priority: int = Field(default=5, ge=0, le=9)
    unitaries: List[UnitaryTarget] = Field(min_length=1, max_length=MAX_TARGETS_PER_JOB)


class JobPayload(BaseModel):
    """What actually goes on the queue, after the server fills in identity."""

    job_id: str
    user_id: str
    lease_token: Optional[str] = None
    priority: int = 5
    unitaries: List[UnitaryTarget]


class LeaseRequest(BaseModel):
    ttl_seconds: Optional[int] = Field(default=None, ge=1)
    note: Optional[str] = Field(default=None, max_length=280)


class LeaseToken(BaseModel):
    token: str


class WhoAmI(BaseModel):
    login: str
    display_name: str
    role: str
    address: str
    permissions: List[str]
    quotas: Dict[str, int]
    jobs_today: int
    jobs_running: int


class ChipStatus(BaseModel):
    """Configured state of the chip.

    There is deliberately no `connected` flag: the device is opened by the
    worker process, so the API cannot observe link state and reporting one
    would be a guess presented as fact.
    """

    backend: str
    grid_size: str
    n_channels: int
    max_current_mA: float
    lease: Optional[Dict] = None
    calibrated_heaters: int
    uncalibrated_heaters: List[str] = Field(default_factory=list)
