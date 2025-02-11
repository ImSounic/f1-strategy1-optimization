import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from main import load_multi_year_data, train_ml_model, simulate_alternative_strategies, get_f1_weather

# Set Streamlit page configuration
st.set_page_config(page_title="F1 Strategy Optimizer", layout="wide")

# 🔥 F1-Inspired Styling
F1_THEME = {
    "background_color": "#181818",
    "text_color": "#ffffff",
    "primary_color": "#ff1e00",
    "team_colors": {
        "Red Bull": "#1E41FF",
        "Ferrari": "#DC0000",
        "Mercedes": "#00D2BE",
        "McLaren": "#FF8700",
        "Aston Martin": "#006F62",
        "Alpine": "#0090FF",
        "Haas": "#FFFFFF",
        "AlphaTauri": "#2B4562",
        "Williams": "#37BEDD",
        "Alfa Romeo": "#900000"
    }
}

# Apply styling
st.markdown(
    f"""
    <style>
        body {{
            background-color: {F1_THEME['background_color']};
            color: {F1_THEME['text_color']};
        }}
        .sidebar .sidebar-content {{
            background-color: #262626;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {F1_THEME['primary_color']};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# 🎯 Sidebar - User Input
st.sidebar.title("🏎️ F1 Strategy Optimizer")
gp_selected = st.sidebar.selectbox("Select Grand Prix", ["Bahrain", "Monza", "Silverstone", "Spa"])
driver_code = st.sidebar.text_input("Enter Driver's Code (e.g., VER, HAM, LEC)", "VER")
grid_position = st.sidebar.number_input("Enter Assumed Grid Position", min_value=1, max_value=20, value=1)

# 🌎 Load Data & Weather
YEARS = [2022, 2023, 2024]
laps, sessions = load_multi_year_data(YEARS, gp_selected)
total_laps = int(laps['LapNumber'].max())
weather = get_f1_weather(sessions[-1])

# Train ML Model
model, feature_names = train_ml_model(laps)

# 🔥 Simulate Strategies
alternative_strategies = simulate_alternative_strategies(model, weather, total_laps, feature_names)

# 📌 Display Strategy Results
st.subheader(f"🏁 **Best Strategy for {gp_selected}**")
for strategy_name, (strategy, race_time, lap_times) in alternative_strategies.items():
    with st.expander(f"🔹 {strategy_name} Strategy"):
        st.markdown(f"**Predicted Race Time:** ⏱️ {str(pd.to_timedelta(race_time, unit='s'))}")
        for stint in strategy:
            st.markdown(f" - **Lap {stint[0]} - {stint[1]}:** {stint[2]}")

# 📊 **Interactive Lap Time Visualization**
st.subheader("📉 Lap Time Visualization")
fig, ax = plt.subplots(figsize=(10, 5))

compound_colors = {"SOFT": "red", "MEDIUM": "yellow", "HARD": "white"}
for strategy_name, (_, _, lap_times) in alternative_strategies.items():
    laps, times, compounds = zip(*lap_times)
    ax.plot(laps, times, linestyle="-", linewidth=1, color="gray", alpha=0.5)
    for compound, color in compound_colors.items():
        lap_subset = [lap for lap, comp in zip(laps, compounds) if comp == compound]
        time_subset = [time for time, comp in zip(times, compounds) if comp == compound]
        ax.scatter(lap_subset, time_subset, color=color, label=compound, s=30)

ax.set_xlabel("Lap Number", fontsize=12)
ax.set_ylabel("Lap Time (s)", fontsize=12)
ax.set_title("Lap Time Trends", fontsize=14)
ax.legend()
ax.grid(color='gray', linestyle='--', linewidth=0.5)

st.pyplot(fig)

# 🔎 **Race History Search**
st.sidebar.subheader("🔍 Search Race History")
search_gp = st.sidebar.text_input("Search Grand Prix (e.g., Monza 2023)", "")

if search_gp:
    # Fake historical race data (to be replaced with real data later)
    past_races = {
        "Monza 2023": {"Winner": "VER", "Fastest Lap": "HAM", "Best Strategy": "1-Stop (MEDIUM → HARD)"},
        "Silverstone 2022": {"Winner": "LEC", "Fastest Lap": "NOR", "Best Strategy": "2-Stop (SOFT → MEDIUM → HARD)"},
        "Spa 2021": {"Winner": "VER", "Fastest Lap": "HAM", "Best Strategy": "1-Stop (HARD → MEDIUM)"},
    }

    if search_gp in past_races:
        st.sidebar.markdown(f"### **{search_gp}**")
        st.sidebar.markdown(f"🏆 **Winner:** {past_races[search_gp]['Winner']}")
        st.sidebar.markdown(f"⚡ **Fastest Lap:** {past_races[search_gp]['Fastest Lap']}")
        st.sidebar.markdown(f"🛠 **Best Strategy:** {past_races[search_gp]['Best Strategy']}")
    else:
        st.sidebar.warning("No data found for this Grand Prix.")

# 🏁 **Final Note**
st.sidebar.write("📢 _Developed by [ImSounic](https://github.com/ImSounic)_")

