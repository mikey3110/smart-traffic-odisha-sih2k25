import requests, time, random
import datetime

URL = "http://127.0.0.1:5001/cv/counts"

def send_counts():
    while True:
        payload = {
            "camera_id": "cam1",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "counts": {
                "car": random.randint(10, 30),
                "bus": random.randint(0, 5),
                "bike": random.randint(5, 20)
            }
        }
        r = requests.post(URL, json=payload)
        print(r.json())
        time.sleep(5)

if __name__ == "__main__":
    send_counts()
