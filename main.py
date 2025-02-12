import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st
import plotly.express as px

# Enable caching for faster data retrieval
fastf1.Cache.enable_cache('cache')


# Function to load multi-year race data dynamically
@st.cache_data
def load_multi_year_data(years, gp_name):
    laps_data = []
    sessions = []
    for year in years:
        session = fastf1.get_session(year, gp_name, 'R')
        session.load()
        sessions.append(session)
        laps_data.append(session.laps)
    return pd.concat(laps_data, ignore_index=True), sessions


# Function to get weather data from FastF1
def get_f1_weather(session):
    weather_data = session.weather_data.dropna()
    if weather_data.empty:
        print("⚠️ Warning: No weather data available. Using default conditions.")
        return {'temperature': 25, 'humidity': 50, 'weather_condition': 'Clear'}
    latest_weather = weather_data.iloc[-1]  # Get the latest recorded weather data

    weather = {
        'temperature': float(latest_weather['AirTemp']),
        'humidity': float(latest_weather['Humidity']),
        'weather_condition': "Clear" if latest_weather['Rainfall'] == 0 else "Rain"
    }
    return weather

# Train and cache ML model
@st.cache_resource
def train_ml_model(laps):
    # Ensure _laps is a DataFrame
    if not isinstance(laps, pd.DataFrame):
        raise ValueError("Expected a Pandas DataFrame for laps data.")

    # Drop rows where 'LapTime' is missing
    laps = laps.dropna(subset=['LapTime']).copy()

    # Convert LapTime to total seconds
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

    # Feature selection & encoding categorical variables
    features = laps[['LapNumber', 'TyreLife', 'TrackStatus', 'Compound']]
    features = pd.get_dummies(features, columns=['Compound', 'TrackStatus'])

    # Target variable
    target = laps['LapTimeSeconds']

    # Train the RandomForest model
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(features, target)

    return model, features.columns




# Function to predict lap times using the trained model
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
            for c in ['HARD', 'MEDIUM', 'SOFT']:
                features[f'Compound_{c}'] = 1 if compound == c else 0

            features_df = pd.DataFrame([features])
            features_df = features_df.reindex(columns=feature_names, fill_value=0)
            lap_time = model.predict(features_df)[0]
            if weather['weather_condition'] == 'Rain':
                lap_time *= 1.2  # Slower lap times in wet conditions
            lap_times.append((lap, lap_time, compound))
    return lap_times


# Function to plot lap times with compounds using both scatter and line plots
def plot_lap_times(lap_times):
    compound_colors = {"SOFT": "red", "MEDIUM": "yellow", "HARD": "white"}

    # Convert data to DataFrame
    df = pd.DataFrame(lap_times, columns=["Lap", "LapTime", "Compound"])

    # Convert LapTime from seconds to MM:SS format for hover
    df["LapTime_MMSS"] = df["LapTime"].apply(lambda x: f"{int(x//60)}:{int(x%60):02d}")

    # Create interactive Plotly scatter plot
    fig = px.scatter(df,
                     x="Lap",
                     y="LapTime",
                     color="Compound",
                     color_discrete_map=compound_colors,
                     hover_data={"Lap": True, "LapTime_MMSS": True},
                     title="Race Lap Times")

    # Customize layout
    fig.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        xaxis=dict(title="Lap Number"),
        yaxis=dict(title="Lap Time (s)"),
        hovermode="x unified"
    )

    # Show plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)



# Function to simulate different strategies
def simulate_strategies(model, weather, total_laps, feature_names):
    compounds = ['SOFT', 'MEDIUM', 'HARD']
    best_strategy = None
    best_time = float('inf')
    best_lap_times = []

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
                race_time = sum(time for _, time, _ in lap_times) + 2 * 20

                if race_time < best_time:
                    best_time = race_time
                    best_strategy = strategy
                    best_lap_times = lap_times

    return best_strategy, best_time, best_lap_times


# Function to simulate multiple strategies
def simulate_alternative_strategies(model, weather, total_laps, feature_names):
    strategies = {
        "Aggressive 2-Stop": [(1, total_laps // 3, "SOFT"), (total_laps // 3 + 1, 2 * total_laps // 3, "SOFT"),
                              (2 * total_laps // 3 + 1, total_laps, "MEDIUM")],
        "Conservative 1-Stop": [(1, total_laps // 2, "HARD"), (total_laps // 2 + 1, total_laps, "MEDIUM")],
        "Balanced 2-Stop": [(1, total_laps // 3, "MEDIUM"), (total_laps // 3 + 1, 2 * total_laps // 3, "HARD"),
                            (2 * total_laps // 3 + 1, total_laps, "SOFT")]
    }
    results = {}

    for strategy_name, strategy in strategies.items():
        lap_times = predict_lap_times(model, strategy, weather, feature_names)
        race_time = sum(time for _, time, _ in lap_times) + len(strategy) * 20
        results[strategy_name] = (strategy, race_time, lap_times)

    return results

# Function to simulate driver-specific strategy
def simulate_driver_strategy(driver_name, grid_position, model, weather, total_laps, feature_names):
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


# Main function
def main():
    GP_NAME = "Monaco"
    YEARS = [2022, 2023, 2024]

    driver_name = input("Enter Driver's Code (e.g., VER, LEC, HAM): ").upper()
    grid_position = int(input(f"Enter assumed grid position for {driver_name}: "))


    laps, sessions = load_multi_year_data(YEARS, GP_NAME)
    total_laps = int(laps['LapNumber'].max())

    # Use the latest session for weather data
    weather = get_f1_weather(sessions[-1])
    print(f"Current Weather: {weather}")

    model, feature_names = train_ml_model(laps)

    best_strategy, best_time, best_lap_times = simulate_strategies(model, weather, total_laps, feature_names)

    alternative_strategies = simulate_alternative_strategies(model, weather, total_laps, feature_names)

    #BEST STRATEGY
    print("\n🏁 **Best Strategy Found:**")
    for stint in best_strategy:
        print(f"Lap {stint[0]} - {stint[1]}: {stint[2]}")
    print(f"\n🔵 **Predicted Race Time:** {str(datetime.timedelta(seconds=int(best_time)))}")

    #ALTERNATIVE STRATEGIES
    for strategy_name, (strategy, race_time, lap_times) in alternative_strategies.items():
        print(f"\n🏁 **{strategy_name}:**")
        for stint in strategy:
            print(f"Lap {stint[0]} - {stint[1]}: {stint[2]}")
        print(f"🔵 **Predicted Race Time:** {str(datetime.timedelta(seconds=int(race_time)))}")
        plot_lap_times(lap_times)

        driver_strategy, driver_race_time, driver_lap_times = simulate_driver_strategy(
            driver_name, grid_position, model, weather, total_laps, feature_names)

        print("\n🏁 **Driver-Specific Strategy:**")
        for stint in driver_strategy:
            print(f"Lap {stint[0]} - {stint[1]}: {stint[2]}")
        print(
            f"\n🔵 **Predicted Race Time for {driver_name}:** {str(datetime.timedelta(seconds=int(driver_race_time)))}")

        plot_lap_times(driver_lap_times)

    # Plot the lap times
    plot_lap_times(best_lap_times)


if __name__ == "__main__":
    main()
