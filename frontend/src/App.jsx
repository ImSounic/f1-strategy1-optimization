import React, { useState } from "react";
import { fetchStrategy } from "./api";
import Plot from "react-plotly.js";
import "./App.css";

const compoundColors = { SOFT: "red", MEDIUM: "yellow", HARD: "white" };

const App = () => {
    const [gpName, setGpName] = useState("Monaco");
    const [driverName, setDriverName] = useState("VER");
    const [gridPosition, setGridPosition] = useState(1);
    const [strategyData, setStrategyData] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async () => {
        setLoading(true);
        const data = await fetchStrategy(gpName, driverName, gridPosition);
        setStrategyData(data);
        setLoading(false);
    };

    return (
        <div className="app-container">
            <h1 className="title">🏎️ F1 Strategy Optimizer</h1>

            <div className="input-container">
                <label>Grand Prix: </label>
                <select value={gpName} onChange={(e) => setGpName(e.target.value)}>
                    <option value="Monaco">Monaco</option>
                    <option value="Monza">Monza</option>
                    <option value="Bahrain">Bahrain</option>
                    <option value="Silverstone">Silverstone</option>
                    <option value="Spa">Spa</option>
                </select>
            </div>

            <div className="input-container">
                <label>Driver Code: </label>
                <input type="text" value={driverName} onChange={(e) => setDriverName(e.target.value.toUpperCase())} />
            </div>

            <div className="input-container">
                <label>Grid Position: </label>
                <input type="number" value={gridPosition} min="1" max="20" onChange={(e) => setGridPosition(Number(e.target.value))} />
            </div>

            <button className="submit-btn" onClick={handleSubmit} disabled={loading}>
                {loading ? "Optimizing..." : "Optimize Strategy"}
            </button>

            {strategyData && (
                <div className="strategy-container">
                    <h2>🏁 Best Strategy</h2>
                    <p><strong>Predicted Race Time:</strong> {strategyData.best_strategy.predicted_time}</p>
                    {strategyData.best_strategy.strategy.map((stint, index) => (
                        <p key={index}>Lap {stint[0]} - {stint[1]}: {stint[2]}</p>
                    ))}

                    <h2>🔄 Alternative Strategies</h2>
                    {Object.entries(strategyData.alternative_strategies).map(([name, strat], index) => (
                        <div key={index} className="strategy-section">
                            <h3>{name}</h3>
                            <p><strong>Predicted Race Time:</strong> {strat.predicted_time}</p>
                            {strat.strategy.map((stint, i) => (
                                <p key={i}>Lap {stint[0]} - {stint[1]}: {stint[2]}</p>
                            ))}
                        </div>
                    ))}

                    <h2>📊 Lap Time Visualization</h2>
                    {strategyData.lap_times && (
                        <Plot
                            data={[
                                {
                                    x: strategyData.lap_times.map(lap => lap.Lap),
                                    y: strategyData.lap_times.map(lap => lap.LapTime),
                                    mode: "lines+markers",
                                    marker: {
                                        color: strategyData.lap_times.map(lap => compoundColors[lap.Compound] || "gray"),
                                        size: 8
                                    },
                                    type: "scatter"
                                }
                            ]}
                            layout={{
                                title: "Race Lap Times",
                                xaxis: { title: "Lap Number" },
                                yaxis: { title: "Lap Time (s)", autorange: "reversed" },
                                plot_bgcolor: "black",
                                paper_bgcolor: "black",
                                font: { color: "white" }
                            }}
                        />
                    )}
                </div>
            )}
        </div>
    );
};

export default App;
