import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Enable caching for faster data retrieval
fastf1.Cache.enable_cache('cache')


# Function to load multi-year race data dynamically
def load_multi_year_data(years, gp_name, driver_name=None):
    laps_data = []
    sessions = []
    for year in years:
        session = fastf1.get_session(year, gp_name, 'R')
        session.load()
        sessions.append(session)
        laps = session.laps
        if driver_name:
            laps = laps[laps['Driver'] == driver_name]
        laps_data.append(laps)
    return pd.concat(laps_data, ignore_index=True), sessions


# Function to get weather data from FastF1
def get_f1_weather(session):
    weather_data = session.weather_data.dropna()
    if weather_data.empty:
        print("⚠️ Warning: No weather data available. Using default conditions.")
        return {'temperature': 25, 'humidity': 50, 'weather_condition': 'Clear'}
    latest_weather = weather_data.iloc[-1]
    return {
        'temperature': float(latest_weather['AirTemp']),
        'humidity': float(latest_weather['Humidity']),
        'weather_condition': "Clear" if latest_weather['Rainfall'] == 0 else "Rain"
    }


# Function to train ML model
def train_ml_model(laps):
    laps = laps.dropna(subset=['LapTime'])
    laps = laps.copy()
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    features = laps[['LapNumber', 'TyreLife', 'TrackStatus', 'Compound']]
    features = pd.get_dummies(features, columns=['Compound', 'TrackStatus'])
    target = laps['LapTimeSeconds']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features, target)
    return model, features.columns


# Function to predict lap times
def predict_lap_times(model, strategy, weather, feature_names):
    lap_times = []
    for stint in strategy:
        start_lap, end_lap, compound = stint
        for lap in range(start_lap, end_lap + 1):
            features = {
                'LapNumber': lap,
                'TyreLife': lap - start_lap + 1,
                'TrackStatus_1': 1
            }

            #TO-DO
            #INSTEAD OF HARD MEDIUM SOFT, USE C1 - C5 FOR BETTER STRATEGY PREDICTION
            for c in ['HARD', 'MEDIUM', 'SOFT']:
                features[f'Compound_{c}'] = 1 if compound == c else 0

            features_df = pd.DataFrame([features])
            features_df = features_df.reindex(columns=feature_names, fill_value=0)
            lap_time = model.predict(features_df)[0]
            if weather['weather_condition'] == 'Rain':
                lap_time *= 1.2
            lap_times.append((lap, lap_time, compound))
    return lap_times


# Function to simulate driver-specific strategy
def simulate_driver_strategy(driver_name, grid_position, model, weather, total_laps, feature_names):
    # TO-DO
    # INSTEAD OF HARD MEDIUM SOFT, USE C1 - C5 FOR BETTER STRATEGY PREDICTION
    compounds = ['SOFT', 'MEDIUM', 'HARD']
    if grid_position <= 5:  # Front row strategy
        strategy = [(1, total_laps // 3, "MEDIUM"), (total_laps // 3 + 1, 2 * total_laps // 3, "HARD"),
                    (2 * total_laps // 3 + 1, total_laps, "SOFT")]
    elif 6 <= grid_position <= 12:  # Midfield strategy
        strategy = [(1, total_laps // 2, "HARD"), (total_laps // 2 + 1, total_laps, "SOFT")]
    else:  # Backmarker strategy
        strategy = [(1, total_laps // 2, "MEDIUM"), (total_laps // 2 + 1, total_laps, "HARD")]

    lap_times = predict_lap_times(model, strategy, weather, feature_names)
    race_time = sum(time for _, time, _ in lap_times) + len(strategy) * 20
    return strategy, race_time, lap_times


# Function to plot lap times with compounds
def plot_lap_times(lap_times, title):
    plt.figure(figsize=(10, 5))
    plt.style.use("dark_background")
    compound_colors = {"SOFT": "red", "MEDIUM": "yellow", "HARD": "white"}

    lap_numbers, lap_time_values, compounds_used = zip(*lap_times)

    plt.plot(lap_numbers, lap_time_values, linestyle='-', linewidth=1, color="gray", alpha=0.5)

    for compound, color in compound_colors.items():
        lap_subset = [lap for lap, comp in zip(lap_numbers, compounds_used) if comp == compound]
        time_subset = [time for time, comp in zip(lap_time_values, compounds_used) if comp == compound]
        plt.scatter(lap_subset, time_subset, color=color, label=compound, s=30)

    def format_seconds_to_mmss(x, _):
        minutes, seconds = divmod(int(x), 60)
        return f"{minutes}:{seconds:02d}"

    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_seconds_to_mmss))
    plt.xlabel("Lap Number", fontsize=12, fontweight='bold', color='black')
    plt.ylabel("Lap Time (MM:SS)", fontsize=12, fontweight='bold', color='black')
    plt.title(title, fontsize=14, fontweight='bold', color='black')
    plt.xticks(color='black')
    plt.yticks(color='black')
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.show()


# Main function
def main():
    GP_NAME = "Brazil"
    YEARS = [2022, 2023, 2024]

    driver_name = input("Enter Driver's Code (e.g., VER, LEC, HAM): ").upper()
    grid_position = int(input(f"Enter assumed grid position for {driver_name}: "))

    laps, sessions = load_multi_year_data(YEARS, GP_NAME, driver_name)
    total_laps = int(laps['LapNumber'].max())
    weather = get_f1_weather(sessions[-1])

    print(f"Current Weather: {weather}")

    model, feature_names = train_ml_model(laps)

    driver_strategy, driver_race_time, driver_lap_times = simulate_driver_strategy(
        driver_name, grid_position, model, weather, total_laps, feature_names)

    print("\n🏁 **Driver-Specific Strategy:**")
    for stint in driver_strategy:
        print(f"Lap {stint[0]} - {stint[1]}: {stint[2]}")
    print(f"\n🔵 **Predicted Race Time for {driver_name}:** {str(datetime.timedelta(seconds=int(driver_race_time)))}")

    plot_lap_times(driver_lap_times, f"{driver_name} Strategy (Starting P{grid_position})")


if __name__ == "__main__":
    main()
