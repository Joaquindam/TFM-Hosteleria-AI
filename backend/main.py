from fastapi import FastAPI

from backend import reservas_store
from backend.routers import events, forecast, noshow, reservations, reviews, sales, weather

app = FastAPI(title="Restaurant Intelligence API")


@app.on_event("startup")
def _startup() -> None:
    reservas_store.init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(forecast.router)
app.include_router(sales.router)
app.include_router(noshow.router)
app.include_router(reservations.router)
app.include_router(weather.router)
app.include_router(events.router)
app.include_router(reviews.router)
