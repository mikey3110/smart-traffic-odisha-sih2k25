from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)
counts_log = []

@app.route("/cv/counts", methods=["POST"])
def cv_counts():
    data = request.json
    if "timestamp" not in data:
        data["timestamp"] = datetime.utcnow().isoformat()
    counts_log.append(data)
    print(f"📥 Received: {data}")
    return jsonify({"status": "ok", "received": data}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
