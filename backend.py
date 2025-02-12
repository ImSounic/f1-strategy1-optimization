from flask import Flask, request, jsonify
from flask_cors import CORS  # Allow frontend requests
import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import datetime

app = Flask(__name__)
CORS(app)

# Enable caching for FastF1
fastf1.Cache.enable_cache('cache')

# Load multi-year data
def load_multi_year_data(years, gp_name):
    laps_data, sessions = [], []
    for year in years:
        session = fastf1.get_session(year, gp_name, 'R')
        session.load()
        sessions.append(session)
        laps_data.append(session.laps)
    return pd.concat(laps_data, ignore_index=True), sessions

# Get latest weather data
def get_f1_weather(session):
    weather_data = session.weather_data.dropna()
    if weather_data.empty:
        return {'temperature': 25, 'humidity': 50, 'weather_condition': 'Clear'}
    latest_weather = weather_data.iloc[-1]
    return {
        'temperature': float(latest_weather['AirTemp']),
        'humidity': float(latest_weather['Humidity']),
        'weather_condition': "Clear" if latest_weather['Rainfall'] == 0 else "Rain"
    }

# Train ML model
def train_ml_model(laps):
    laps = laps.dropna(subset=['LapTime']).copy()
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    features = pd.get_dummies(laps[['LapNumber', 'TyreLife', 'TrackStatus', 'Compound']], columns=['Compound', 'TrackStatus'])
    target = laps['LapTimeSeconds']
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(features, target)
    return model, features.columns

# Predict lap times
def predict_lap_times(model, strategy, weather, feature_names):
    lap_times = []
    for stint in strategy:
        start_lap, end_lap, compound = stint
        for lap in range(start_lap, end_lap + 1):
            features = {'LapNumber': lap, 'TyreLife': lap - start_lap + 1, 'TrackStatus_1': 1}
            for c in ['HARD', 'MEDIUM', 'SOFT']:
                features[f'Compound_{c}'] = 1 if compound == c else 0
            features_df = pd.DataFrame([features]).reindex(columns=feature_names, fill_value=0)
            lap_time = model.predict(features_df)[0]
            if weather['weather_condition'] == 'Rain':
                lap_time *= 1.2
            lap_times.append((lap, lap_time, compound))
    return lap_times

# Find best strategy
def find_best_strategy(model, weather, total_laps, feature_names):
    compounds = ['SOFT', 'MEDIUM', 'HARD']
    best_strategy, best_time, best_lap_times = None, float('inf'), []

    for first_tire in compounds:
        for second_tire in [t for t in compounds if t != first_tire]:
            for third_tire in [t for t in compounds if t != second_tire]:
                first_stint_end = min(20, total_laps // 3)
                second_stint_end = min(first_stint_end + 20, 2 * total_laps // 3)
                third_stint_end = total_laps

                strategy = [(1, first_stint_end, first_tire),
                            (first_stint_end + 1, second_stint_end, second_tire),
                            (second_stint_end + 1, third_stint_end, third_tire)]

                lap_times = predict_lap_times(model, strategy, weather, feature_names)
                race_time = sum(time for _, time, _ in lap_times) + 2 * 20  # Adding pit stop time

                if race_time < best_time:
                    best_time, best_strategy, best_lap_times = race_time, strategy, lap_times

    return best_strategy, best_time, best_lap_times

# Generate alternative strategies
def generate_alternative_strategies(model, weather, total_laps, feature_names):
    strategies = {
        "Aggressive 2-Stop": [(1, total_laps // 3, "SOFT"), (total_laps // 3 + 1, 2 * total_laps // 3, "SOFT"),
                              (2 * total_laps // 3 + 1, total_laps, "MEDIUM")],
        "Conservative 1-Stop": [(1, total_laps // 2, "HARD"), (total_laps // 2 + 1, total_laps, "MEDIUM")],
        "Balanced 2-Stop": [(1, total_laps // 3, "MEDIUM"), (total_laps // 3 + 1, 2 * total_laps // 3, "HARD"),
                            (2 * total_laps // 3 + 1, total_laps, "SOFT")]
    }
    results = {}
    for name, strategy in strategies.items():
        lap_times = predict_lap_times(model, strategy, weather, feature_names)
        race_time = sum(time for _, time, _ in lap_times) + len(strategy) * 20
        results[name] = {'strategy': strategy, 'predicted_time': str(datetime.timedelta(seconds=int(race_time)))}
    return results

# Generate driver-specific strategy
def generate_driver_strategy(driver_name, grid_position, model, weather, total_laps, feature_names):
    if grid_position <= 5:
        strategy = [(1, total_laps // 3, "MEDIUM"), (total_laps // 3 + 1, 2 * total_laps // 3, "HARD"),
                    (2 * total_laps // 3 + 1, total_laps, "SOFT")]
    elif 6 <= grid_position <= 12:
        strategy = [(1, total_laps // 2, "HARD"), (total_laps // 2 + 1, total_laps, "SOFT")]
    else:
        strategy = [(1, total_laps // 2, "MEDIUM"), (total_laps // 2 + 1, total_laps, "HARD")]

    lap_times = predict_lap_times(model, strategy, weather, feature_names)
    race_time = sum(time for _, time, _ in lap_times) + len(strategy) * 20
    return {'strategy': strategy, 'predicted_time': str(datetime.timedelta(seconds=int(race_time)))}

# API Endpoints
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    gp_name = data.get('gp_name', 'Monaco')
    driver_name = data.get('driver_name', 'VER')
    grid_position = int(data.get('grid_position', 1))
    years = [2022, 2023, 2024]

    # Load data
    laps, sessions = load_multi_year_data(years, gp_name)
    total_laps = int(laps['LapNumber'].max())
    weather = get_f1_weather(sessions[-1])
    model, feature_names = train_ml_model(laps)

    # Find strategies
    best_strategy, best_time, best_lap_times = find_best_strategy(model, weather, total_laps, feature_names)
    alternative_strategies = generate_alternative_strategies(model, weather, total_laps, feature_names)
    driver_strategy = generate_driver_strategy(driver_name, grid_position, model, weather, total_laps, feature_names)

    # Return response with lap time data
    return jsonify({
        'best_strategy': {'strategy': best_strategy, 'predicted_time': str(datetime.timedelta(seconds=int(best_time)))},
        'alternative_strategies': alternative_strategies,
        'driver_strategy': driver_strategy,
        'lap_times': [{'Lap': lap, 'LapTime': time, 'Compound': compound} for lap, time, compound in best_lap_times]
    })


# Run the API
if __name__ == '__main__':
    app.run(debug=True)
