from pydantic import BaseModel, Field
from typing import List, Dict, Any

class HeaterCurrents(BaseModel):
    heater_theta: float
    heater_phi: float

class UnitaryTarget(BaseModel):
    name: str
    matrix_real: List[List[float]]
    matrix_imag: List[List[float]]
    initial_currents_ma: Dict[str, HeaterCurrents]

class JobPayload(BaseModel):
    job_id: str
    user_id: str
    priority: int = Field(default=5)
    unitaries: List[UnitaryTarget]
