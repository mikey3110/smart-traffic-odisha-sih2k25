from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Dict

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

fake_db: Dict[str, dict] = {}
fake_tokens = {"testtoken": "testuser"}

class TrafficData(BaseModel):
    id: int
    location: str
    vehicles: int
    timestamp: str  # you can also use datetime here if you prefer

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/auth/login")
def login(username: str = "testuser", password: str = "testpass"):
    if username == "testuser" and password == "testpass":
        return {"token": "testtoken"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/traffic/ingest")
async def traffic_ingest(data: TrafficData, token: str = Depends(oauth2_scheme)):
    user = fake_tokens.get(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    fake_db[data.id] = data.dict()

    return {"message": "Data ingested successfully"}

@app.get("/traffic/status/{id}")
def traffic_status(id: int, token: str = Depends(oauth2_scheme)):
    user = fake_tokens.get(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    data = fake_db.get(id)
    if not data:
        raise HTTPException(status_code=404, detail="Data not found")
    return data

@app.get("/traffic/all")
def get_all_traffic(token: str = Depends(oauth2_scheme)):
    user = fake_tokens.get(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return list(fake_db.values())
@app.put("/traffic/update/{id}")
async def update_traffic(id: int, data: TrafficData, token: str = Depends(oauth2_scheme)):
    user = fake_tokens.get(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    if id not in fake_db:
        raise HTTPException(status_code=404, detail="Data not found")
    
    fake_db[id] = data.dict()
    return {"message": "Traffic data updated successfully"}


