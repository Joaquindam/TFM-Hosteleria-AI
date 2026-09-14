class UnsupportedDateError(Exception):
    """La fecha pedida no se puede predecir con los datos e inputs disponibles.

    Se lanza en vez de fabricar una prediccion para una fecha demasiado lejos
    en el futuro (mas de un dia-servicio despues del ultimo dato conocido) o
    cuando faltan overrides obligatorios (meteo) para el unico dia futuro
    soportado. Ver PROJECT_BRIEF.md seccion 4.
    """


class MissingInputError(Exception):
    """Faltan datos obligatorios (p.ej. meteo) para construir la prediccion."""
