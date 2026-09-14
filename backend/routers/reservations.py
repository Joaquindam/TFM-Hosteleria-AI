from datetime import date as date_type

from fastapi import APIRouter, HTTPException

from backend import reservas_store
from backend.schemas import ReservaCreate, ReservaUpdate

router = APIRouter(prefix="/reservations", tags=["reservations"])


@router.get("/{target_date}")
def get_reservations(target_date: date_type):
    reservas = reservas_store.list_reservas_by_date(target_date.isoformat(), include_cancelled=False)
    return {
        "date": target_date.isoformat(),
        "reservas": reservas,
        "total_reservas": len(reservas),
        "total_comensales": sum(r["comensales"] for r in reservas),
        "source": "app",
    }


@router.post("", status_code=201)
def create_reservation(payload: ReservaCreate):
    return reservas_store.create_reserva(
        fecha=payload.fecha.isoformat(),
        comensales=payload.comensales,
        hora=payload.hora,
        nombre=payload.nombre,
        notas=payload.notas,
    )


@router.put("/{reserva_id}")
def update_reservation(reserva_id: int, payload: ReservaUpdate):
    fields = payload.model_dump(exclude_unset=True)
    if fields.get("fecha") is not None:
        fields["fecha"] = fields["fecha"].isoformat()
    try:
        return reservas_store.update_reserva(reserva_id, **fields)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{reserva_id}")
def cancel_reservation(reserva_id: int):
    """Borrado logico (estado='cancelada'), nunca DELETE fisico -- ver CLAUDE.md."""
    try:
        return reservas_store.cancel_reserva(reserva_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
