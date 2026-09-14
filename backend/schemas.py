from datetime import date as date_type
from typing import Optional

from pydantic import BaseModel, Field


class ReservaCreate(BaseModel):
    fecha: date_type
    comensales: int = Field(gt=0)
    hora: Optional[str] = None
    nombre: Optional[str] = None
    notas: Optional[str] = None


class ReservaUpdate(BaseModel):
    fecha: Optional[date_type] = None
    hora: Optional[str] = None
    comensales: Optional[int] = Field(default=None, gt=0)
    nombre: Optional[str] = None
    estado: Optional[str] = None
    notas: Optional[str] = None


class WeatherOverrides(BaseModel):
    temperature_max: float
    temperature_min: float
    temperature_mean: float
    precipitation_mm: float
    precipitation_hours: float
    wind_speed_max: float
    sunshine_duration_h: float


class NoShowRequest(BaseModel):
    reservation_date: date_type
    shift: str  # "Comida" | "Cena"
    people: int = Field(gt=0)
    origin: str  # "appmovil" | "moduloweb" | "software" | "terceros" (no "walk in": el modelo nunca lo vio)
    es_grupo_grande: bool = False
    zone: Optional[str] = None  # "Sala" | "Terraza Cubierta"; por defecto "Sala"
    antelacion_horas: float = Field(gt=0)
    reservas_mismo_dia_turno: Optional[int] = None


class LLMAppChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class LLMAppChatRequest(BaseModel):
    message: str
    history: list[LLMAppChatMessage] = Field(default_factory=list)
