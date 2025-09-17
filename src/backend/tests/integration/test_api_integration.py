import pytest
import datetime
from httpx import AsyncClient
from httpx._transports.asgi import ASGITransport
from app.main import app

# ---------------- Health Check ----------------
@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# ---------------- Authentication Tests ----------------
@pytest.mark.asyncio
async def test_login_success():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/auth/login", params={"username": "testuser", "password": "testpass"})
    assert response.status_code == 200
    assert response.json() == {"token": "testtoken"}

@pytest.mark.asyncio
async def test_login_failure():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/auth/login", params={"username": "wrong", "password": "wrong"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid credentials"}

# ---------------- Traffic Ingestion Tests ----------------




@pytest.mark.asyncio
async def test_traffic_ingest_invalid_payload():
    payload = {"id": 1, "location": "Main Street"}  # Missing 'vehicles' and 'timestamp'
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/traffic/ingest", json=payload, headers={"Authorization": "Bearer testtoken"})
    assert response.status_code == 422  # Validation error

@pytest.mark.asyncio
async def test_traffic_ingest_unauthorized():
    payload = {
        "id": 1,
        "location": "Main Street",
        "vehicles": 25,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/traffic/ingest", json=payload)  # No token
    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}

# ---------------- Traffic Status Tests ----------------


@pytest.mark.asyncio
async def test_traffic_status_not_found():
    traffic_id = 9999  # Non-existing ID
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get(f"/traffic/status/{traffic_id}", headers={"Authorization": "Bearer testtoken"})
    assert response.status_code == 404
    assert response.json() == {"detail": "Data not found"}  # <-- updated


@pytest.mark.asyncio
async def test_traffic_status_unauthorized():
    traffic_id = 1
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get(f"/traffic/status/{traffic_id}")  # No token
    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}
