from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import xgboost as xgb

# Load the trained XGBoost model
model = joblib.load("xgboost_model.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define the input schema
class CarFeatures(BaseModel):
    Engine_volume: float
    Mileage: float
    Cylinders: int
    Doors: int
    Airbags: int
    Turbo: bool
    Age: int
    Drive_4x4: bool
    Drive_Front: bool
    Drive_Rear: bool
    Gear_Automatic: bool
    Gear_Manual: bool
    Gear_Tiptronic: bool
    Gear_Variator: bool
    Fuel_CNG: bool
    Fuel_Diesel: bool
    Fuel_Hybrid: bool
    Fuel_Hydrogen: bool
    Fuel_LPG: bool
    Fuel_Petrol: bool
    Fuel_Plug_in_Hybrid: bool  # Renamed for Python variable compatibility
    Leather_No: bool
    Leather_Yes: bool

@app.post("/predict/")
def predict(input_data: CarFeatures):
    try:
        # Convert input to a NumPy array for XGBoost
        input_features = np.array([[
            input_data.Engine_volume, input_data.Mileage, input_data.Cylinders,
            input_data.Doors, input_data.Airbags, int(input_data.Turbo), input_data.Age,
            int(input_data.Drive_4x4), int(input_data.Drive_Front), int(input_data.Drive_Rear),
            int(input_data.Gear_Automatic), int(input_data.Gear_Manual),
            int(input_data.Gear_Tiptronic), int(input_data.Gear_Variator),
            int(input_data.Fuel_CNG), int(input_data.Fuel_Diesel), int(input_data.Fuel_Hybrid),
            int(input_data.Fuel_Hydrogen), int(input_data.Fuel_LPG), int(input_data.Fuel_Petrol),
            int(input_data.Fuel_Plug_in_Hybrid), int(input_data.Leather_No), int(input_data.Leather_Yes)
        ]]).astype(float)  # Ensure everything is in float format for XGBoost

        # Make a prediction using XGBoost
        prediction = model.predict(xgb.DMatrix(input_features))

        return {"predicted_value": float(prediction[0])}

    except Exception as e:
        return {"error": str(e)}
