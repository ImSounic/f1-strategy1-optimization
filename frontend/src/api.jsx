import axios from 'axios';

export const fetchStrategy = async (gpName, driverName, gridPosition) => {
    try {
        const response = await axios.post("http://127.0.0.1:5000/predict", {
            gp_name: gpName,
            driver_name: driverName,
            grid_position: gridPosition
        });
        return response.data;
    } catch (error) {
        console.error("Error fetching strategy:", error);
        return null;
    }
};
