import streamlit as st
import datetime
from main import (
    load_multi_year_data, get_f1_weather, train_ml_model,
    simulate_strategies, simulate_alternative_strategies,
    simulate_driver_strategy, plot_lap_times
)

# Set page title
st.set_page_config(page_title="F1 Strategy Optimization", layout="wide")

# Sidebar
st.sidebar.title("F1 Strategy Optimizer")
gp_name = st.sidebar.selectbox("Select Grand Prix", ["Monaco", "Monza", "Bahrain", "Silverstone", "Spa"])
years = [2022, 2023, 2024]  # Hardcoded for now

st.sidebar.subheader("Driver Strategy Settings")
driver_name = st.sidebar.text_input("Enter Driver's Code (e.g., VER, HAM, LEC)", "VER")
grid_position = st.sidebar.number_input("Enter Assumed Grid Position", min_value=1, max_value=20, value=1)

# Load data
st.write(f"### Loading Data for {gp_name} Grand Prix")
laps, sessions = load_multi_year_data(years, gp_name)
total_laps = int(laps['LapNumber'].max())
weather = get_f1_weather(sessions[-1])

st.write(f"**Current Weather:** 🌡️ {weather['temperature']}°C, 💧 {weather['humidity']}%, ☁️ {weather['weather_condition']}")
# Ensure laps is a Pandas DataFrame
laps_df = laps.to_dataframe() if hasattr(laps, 'to_dataframe') else laps

# Train the model (cached)
model, feature_names = train_ml_model(laps_df)

# Best Strategy
st.write("## 🏁 Best Strategy Found")
best_strategy, best_time, best_lap_times = simulate_strategies(model, weather, total_laps, feature_names)
st.write(f"**Predicted Race Time:** ⏱ {str(datetime.timedelta(seconds=int(best_time)))}")
for stint in best_strategy:
    st.write(f"- **Lap {stint[0]} - {stint[1]}:** {stint[2]}")

# Alternative Strategies
st.write("## 🔄 Alternative Strategies")
alternative_strategies = simulate_alternative_strategies(model, weather, total_laps, feature_names)
for strat_name, (strategy, race_time, _) in alternative_strategies.items():
    with st.expander(f"🔹 {strat_name} Strategy"):
        st.write(f"**Predicted Race Time:** ⏱ {str(datetime.timedelta(seconds=int(race_time)))}")
        for stint in strategy:
            st.write(f"- **Lap {stint[0]} - {stint[1]}:** {stint[2]}")

# Driver-Specific Strategy
st.write(f"## 🚗 {driver_name}'s Personalized Strategy (Grid Position: {grid_position})")
driver_strategy, driver_time, driver_lap_times = simulate_driver_strategy(
    driver_name, grid_position, model, weather, total_laps, feature_names
)
st.write(f"**Predicted Race Time:** ⏱ {str(datetime.timedelta(seconds=int(driver_time)))}")
for stint in driver_strategy:
    st.write(f"- **Lap {stint[0]} - {stint[1]}:** {stint[2]}")

# Show graph
st.write("## 📊 Lap Time Visualization")
plot_lap_times(best_lap_times)
