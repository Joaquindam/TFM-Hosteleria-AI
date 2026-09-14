"""Almacen de reservas (SQLite), rellenado a mano desde la app -- ver
PROJECT_BRIEF.md seccion 5.1 (entrada manual, no integracion externa) y
seccion 5.2 / CLAUDE.md (persistencia acumulativa: nunca DELETE fisico).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, date as date_type

from ml.paths import DATA_DIR

DB_PATH = DATA_DIR / "gold" / "reservas.db"

ESTADOS_VALIDOS = {"pendiente", "confirmada", "cancelada", "no_show", "completada"}


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reservas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha TEXT NOT NULL,
                hora TEXT,
                comensales INTEGER NOT NULL,
                nombre TEXT,
                estado TEXT NOT NULL DEFAULT 'pendiente',
                notas TEXT,
                creado_en TEXT NOT NULL,
                actualizado_en TEXT NOT NULL
            )
            """
        )


def create_reserva(fecha: str, comensales: int, hora: str | None = None,
                    nombre: str | None = None, notas: str | None = None) -> dict:
    if comensales <= 0:
        raise ValueError("comensales debe ser mayor que 0")
    date_type.fromisoformat(fecha)  # valida formato YYYY-MM-DD

    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO reservas (fecha, hora, comensales, nombre, estado, notas, creado_en, actualizado_en) "
            "VALUES (?, ?, ?, ?, 'pendiente', ?, ?, ?)",
            (fecha, hora, comensales, nombre, notas, now, now),
        )
        reserva_id = cur.lastrowid
    return get_reserva(reserva_id)


def get_reserva(reserva_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM reservas WHERE id = ?", (reserva_id,)).fetchone()
        return dict(row) if row else None


def list_reservas_by_date(fecha: str, include_cancelled: bool = True) -> list[dict]:
    with _connect() as conn:
        if include_cancelled:
            rows = conn.execute(
                "SELECT * FROM reservas WHERE fecha = ? ORDER BY hora", (fecha,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM reservas WHERE fecha = ? AND estado != 'cancelada' ORDER BY hora",
                (fecha,),
            ).fetchall()
        return [dict(r) for r in rows]


def update_reserva(reserva_id: int, **fields) -> dict:
    allowed = {"fecha", "hora", "comensales", "nombre", "estado", "notas"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        raise ValueError("No hay campos validos que actualizar")
    if "estado" in updates and updates["estado"] not in ESTADOS_VALIDOS:
        raise ValueError(f"estado invalido: {updates['estado']!r} (validos: {sorted(ESTADOS_VALIDOS)})")
    if "comensales" in updates and updates["comensales"] <= 0:
        raise ValueError("comensales debe ser mayor que 0")

    updates["actualizado_en"] = datetime.utcnow().isoformat()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    with _connect() as conn:
        cur = conn.execute(f"UPDATE reservas SET {set_clause} WHERE id = ?", (*updates.values(), reserva_id))
        if cur.rowcount == 0:
            raise ValueError(f"No existe la reserva {reserva_id}")
    return get_reserva(reserva_id)


def cancel_reserva(reserva_id: int) -> dict:
    """Borrado logico -- nunca DELETE fisico (ver CLAUDE.md, PROJECT_BRIEF.md 5.2)."""
    return update_reserva(reserva_id, estado="cancelada")
