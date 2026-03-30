# ============================================================
#  Solar Tracker — Flask Backend
#  Now includes: /predict  (Open-Meteo + Render Prophet service)
# ============================================================
print("RUNNING THIS FILE >>>>>>>>>>>>>>>")
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import requests
from datetime import datetime
import time

app = Flask(__name__)
CORS(app)


DB_PATH = "solar_tracker.db"

# ── Config ────────────────────────────────────────────────────
LOCATION_LAT   = 12.9716
LOCATION_LON   = 80.2209

# ↓ Replace with your Render service URL after deploying prediction_service.py
RENDER_PREDICT_URL = "https://solar-predictor-service.onrender.com/predict"

# Open-Meteo — free, no API key needed
OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={LOCATION_LAT}&longitude={LOCATION_LON}"
    "&daily=shortwave_radiation_sum,temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean"
    "&forecast_days=16"
    "&timezone=Asia%2FKolkata"
)

# Prediction cache — avoids hammering Render free tier
_predict_cache = {}
CACHE_TTL_SEC  = 3600   # 1 hour

# ── DB helpers ────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                direction     TEXT    NOT NULL,
                angle         INTEGER NOT NULL,
                ldr_east      INTEGER NOT NULL,
                ldr_west      INTEGER NOT NULL,
                voltage_v     REAL    NOT NULL,
                power_mw      REAL    NOT NULL,
                weather       TEXT,
                temp_c        REAL,
                humidity_pct  INTEGER,
                wind_kmh      REAL
            )
        """)
        conn.commit()
    print("[DB] Initialised:", DB_PATH)

# ── Existing routes ───────────────────────────────────────────
@app.route("/")
def home():
    return "Solar Tracker Backend Running"

@app.route("/data", methods=["POST"])
def receive_data():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Invalid JSON"}), 400

    required = ["direction", "angle", "ldr_east", "ldr_west", "voltage_v", "power_mw"]
    missing  = [f for f in required if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    with get_db() as conn:
        conn.execute("""
            INSERT INTO readings
              (timestamp, direction, angle, ldr_east, ldr_west,
               voltage_v, power_mw, weather, temp_c, humidity_pct, wind_kmh)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            payload["direction"],
            int(payload["angle"]),
            int(payload["ldr_east"]),
            int(payload["ldr_west"]),
            float(payload["voltage_v"]),
            float(payload["power_mw"]),
            payload.get("weather",      "Unknown"),
            payload.get("temp_c",       0.0),
            payload.get("humidity_pct", 0),
            payload.get("wind_kmh",     0.0),
        ))
        conn.commit()

    print(f"[POST] {timestamp}  dir={payload['direction']}  "
          f"V={payload['voltage_v']:.2f}V  P={payload['power_mw']:.1f}mW")

    return jsonify({"status": "ok", "timestamp": timestamp}), 201

@app.route("/data", methods=["GET"])
def get_data():
    limit = min(request.args.get("limit", 100, type=int), 500)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM readings ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    readings = [dict(row) for row in rows]
    readings.reverse()
    times  = [r["timestamp"] for r in readings]
    powers = [r["power_mw"]  for r in readings]
    latest = readings[-1] if readings else {}
    return jsonify({
        "latest":  latest,
        "history": readings,
        "chart": {"labels": times, "power": powers}
    }), 200

@app.route("/data/latest", methods=["GET"])
def get_latest():
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM readings ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if not row:
        return jsonify({"error": "No data yet"}), 404
    return jsonify(dict(row)), 200

@app.route("/data/stats", methods=["GET"])
def get_stats():
    with get_db() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)                                  AS total_readings,
                ROUND(MAX(power_mw),   2)                 AS peak_power_mw,
                ROUND(AVG(power_mw),   2)                 AS avg_power_mw,
                ROUND(MAX(voltage_v),  2)                 AS peak_voltage_v,
                ROUND(AVG(voltage_v),  2)                 AS avg_voltage_v,
                ROUND(SUM(power_mw * 0.00833 / 1000), 6) AS energy_wh
            FROM readings
        """).fetchone()
    return jsonify(dict(row)), 200

# NEW: GET /predict
# NEW: GET /predict?days=7 or ?days=30
@app.route("/predict", methods=["GET"])
def get_prediction():
    days = request.args.get("days", 7, type=int)
    days = 30 if days > 7 else 7

    cache_key = f"pred_{days}"
    cached = _predict_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < CACHE_TTL_SEC:
        print(f"[Predict] Serving {days}-day forecast from cache")
        return jsonify(cached["data"]), 200

    with get_db() as conn:
        rows = conn.execute(
            "SELECT timestamp, power_mw, temp_c, humidity_pct FROM readings ORDER BY id DESC LIMIT 200"
        ).fetchall()

    if len(rows) < 10:
        return jsonify({"error": "Not enough data yet. Need at least 10 readings."}), 422

    history = [dict(r) for r in rows]
    history.reverse()

    forecast_weather = []
    try:
        resp = requests.get(OPEN_METEO_URL, timeout=10)
        if resp.status_code == 200:
            om = resp.json()
            daily = om.get("daily", {})
            dates = daily.get("time", [])
            radiation = daily.get("shortwave_radiation_sum", [])
            temp_max = daily.get("temperature_2m_max", [])
            temp_min = daily.get("temperature_2m_min", [])
            humidity = daily.get("relative_humidity_2m_mean", [])
            for i, date in enumerate(dates):
                forecast_weather.append({
                    "date": date,
                    "solar_radiation": radiation[i] if i < len(radiation) else None,
                    "temp_c": round((temp_max[i] + temp_min[i]) / 2, 1)
                               if i < len(temp_max) and i < len(temp_min) else None,
                    "humidity_pct": humidity[i] if i < len(humidity) else None,
                })
    except Exception as e:
        print(f"[Predict] Open-Meteo error: {e}")

    try:
        render_resp = requests.post(
            RENDER_PREDICT_URL,
            json={"history": history, "forecast_weather": forecast_weather, "days": days},
            timeout=60
        )
        if render_resp.status_code != 200:
            return jsonify({
                "error": f"Prediction service returned {render_resp.status_code}",
                "detail": render_resp.text
            }), 502

        result = render_resp.json()
        _predict_cache[cache_key] = {"ts": time.time(), "data": result}
        return jsonify(result), 200

    except requests.exceptions.Timeout:
        return jsonify({"error": "Prediction service timed out. Try again in 30s."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
init_db()
if __name__ == "__main__":
    init_db()
    print("[Server] Running on http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
