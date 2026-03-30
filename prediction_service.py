# ============================================================
#  Solar Tracker — Prophet Prediction Microservice
#  Deploy this on Render as a separate web service.
#
#  Requirements (add to requirements.txt on Render):
#    flask
#    flask-cors
#    prophet
#    pandas
#    numpy
#    gunicorn
#
#  Render start command: gunicorn prediction_service:app
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from prophet import Prophet
import traceback
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Health check ──────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({"status": "Solar Prediction Service Running", "version": "1.0"})

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# ── POST /predict ─────────────────────────────────────────────
# Expected request body:
# {
#   "history": [
#     {
#       "timestamp":  "2024-01-01T10:00:00Z",
#       "power_mw":   850.5,
#       "temp_c":     31.2,
#       "humidity_pct": 60,
#       "solar_radiation": 420.0   ← from Open-Meteo (W/m²)
#     }, ...
#   ],
#   "forecast_weather": [
#     {
#       "date":             "2024-01-08",
#       "solar_radiation":  380.0,
#       "temp_c":           30.5,
#       "humidity_pct":     65
#     }, ...
#   ],
#   "days": 30   ← 7 or 30
# }
@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json(silent=True)
        if not body:
            return jsonify({"error": "Invalid JSON body"}), 400

        history          = body.get("history", [])
        forecast_weather = body.get("forecast_weather", [])
        days             = int(body.get("days", 7))

        if len(history) < 10:
            return jsonify({"error": "Need at least 10 historical readings to predict"}), 422

        # ── Build training dataframe ──────────────────────────
        df = pd.DataFrame(history)
        df["ds"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        df["y"]  = pd.to_numeric(df["power_mw"], errors="coerce").fillna(0)

        # Optional regressors — only include if present in history
        has_temp     = "temp_c"           in df.columns and df["temp_c"].notna().any()
        has_humidity = "humidity_pct"     in df.columns and df["humidity_pct"].notna().any()
        has_solar    = "solar_radiation"  in df.columns and df["solar_radiation"].notna().any()

        # Aggregate to daily averages (Prophet works best on daily data)
        df["ds"] = df["ds"].dt.normalize()
        agg = {"y": "mean"}
        if has_temp:     agg["temp_c"]           = "mean"
        if has_humidity: agg["humidity_pct"]      = "mean"
        if has_solar:    agg["solar_radiation"]   = "mean"

        daily = df.groupby("ds").agg(agg).reset_index()
        daily = daily.sort_values("ds").reset_index(drop=True)

        log.info(f"Training on {len(daily)} daily data points, forecasting {days} days")

        # ── Build Prophet model ───────────────────────────────
        m = Prophet(
            yearly_seasonality=False,   # not enough data yet
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            interval_width=0.80,
        )

        if has_temp     and len(daily) >= 14: m.add_regressor("temp_c")
        if has_humidity and len(daily) >= 14: m.add_regressor("humidity_pct")
        if has_solar    and len(daily) >= 14: m.add_regressor("solar_radiation")

        m.fit(daily)

        # ── Build future dataframe ────────────────────────────
        future = m.make_future_dataframe(periods=days, freq="D")

        # Merge forecast weather into future df
        if forecast_weather:
            fw = pd.DataFrame(forecast_weather)
            fw["ds"] = pd.to_datetime(fw["date"]).dt.normalize()
            future = future.merge(fw[["ds"] + [c for c in ["temp_c","humidity_pct","solar_radiation"] if c in fw.columns]],
                                  on="ds", how="left")
            # Fill historical rows from training data
            for col in ["temp_c","humidity_pct","solar_radiation"]:
                if col in daily.columns and col in future.columns:
                    future[col] = future[col].fillna(daily.set_index("ds")[col].reindex(future["ds"]).values)
                    future[col] = future[col].fillna(method="ffill").fillna(daily[col].mean())

        # Fill any remaining NaN regressors with mean
        for col in ["temp_c","humidity_pct","solar_radiation"]:
            if col in future.columns:
                future[col] = future[col].fillna(daily[col].mean() if col in daily.columns else 0)

        forecast = m.predict(future)

        # ── Extract only future predictions ───────────────────
        last_history_date = daily["ds"].max()
        pred = forecast[forecast["ds"] > last_history_date].copy()
        pred = pred.head(days)

        # Clamp negatives (solar can't be negative)
        pred["yhat"]       = pred["yhat"].clip(lower=0)
        pred["yhat_lower"] = pred["yhat_lower"].clip(lower=0)
        pred["yhat_upper"] = pred["yhat_upper"].clip(lower=0)

        predictions = []
        for _, row in pred.iterrows():
            predictions.append({
                "date":        row["ds"].strftime("%Y-%m-%d"),
                "power_mw":    round(float(row["yhat"]),       1),
                "lower_mw":    round(float(row["yhat_lower"]), 1),
                "upper_mw":    round(float(row["yhat_upper"]), 1),
            })

        # ── Summary stats ──────────────────────────────────────
        powers = [p["power_mw"] for p in predictions]
        best   = max(predictions, key=lambda x: x["power_mw"])
        worst  = min(predictions, key=lambda x: x["power_mw"])
        # Energy: assume each daily avg runs ~6 solar hours → mWh
        total_energy_mwh = round(sum(powers) * 6, 1)

        return jsonify({
            "status":          "ok",
            "days":            days,
            "data_points_used": len(daily),
            "predictions":     predictions,
            "summary": {
                "avg_power_mw":    round(float(np.mean(powers)), 1),
                "peak_power_mw":   round(float(np.max(powers)),  1),
                "min_power_mw":    round(float(np.min(powers)),  1),
                "total_energy_mwh": total_energy_mwh,
                "best_day":        best,
                "worst_day":       worst,
            }
        }), 200

    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
