import sys
import json
import numpy as np

def run_forecast():
    try:
        raw_input = sys.stdin.read()
        if not raw_input:
            print(json.dumps({"error": "No input received"}))
            return

        data = json.loads(raw_input)
        prices = np.array(data["prices"], dtype=np.float32)

        if len(prices) < 32:
            print(json.dumps({"error": "At least 32 price bars required"}))
            return

        last_price = float(prices[-1])
        returns = np.diff(np.log(prices))
        volatility = float(np.std(returns[-14:]))
        short_drift = float(np.mean(returns[-5:]))
        long_drift = float(np.mean(returns[-20:]))

        horizon = 5
        dt = np.arange(1, horizon + 1)
        decay_drift = short_drift * np.exp(-0.15 * dt) + 0.3 * long_drift
        
        p50 = last_price * np.exp(np.cumsum(decay_drift))
        cone_spread = 1.645 * volatility * np.sqrt(dt)
        p10 = p50 * np.exp(-cone_spread)
        p90 = p50 * np.exp(cone_spread)

        result = {
            "status": "ok",
            "last_price": last_price,
            "horizon": horizon,
            "forecast_p50": [round(float(x), 2) for x in p50],
            "lower_bound_p10": [round(float(x), 2) for x in p10],
            "upper_bound_p90": [round(float(x), 2) for x in p90],
            "volatility_score": round(volatility * 100, 3)
        }
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    run_forecast()
