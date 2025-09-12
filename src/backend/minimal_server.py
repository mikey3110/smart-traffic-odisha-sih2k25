from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/traffic/status/{intersection_id}")
def get_traffic_status(intersection_id: str):
    return {
        "intersection_id": intersection_id,
        "status": "active",
        "vehicles": 15,
        "waiting_time": 45
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
