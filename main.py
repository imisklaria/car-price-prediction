import xgboost as xgb
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load trained XGBoost model
model = xgb.Booster()
model.load_model("model.xgb")

# Print feature names after loading
print("Feature names in API-loaded model:", model.feature_names)

app = FastAPI(
    title="Car Price Prediction API",
    description="Predict car prices based on engine specifications and features using a trained XGBoost model.",
    version="1.0.0"
)

# Define API Input Schema
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
    Fuel_Plug_in_Hybrid: bool
    Leather_No: bool
    Leather_Yes: bool

@app.post("/predict/")
def predict(input_data: CarFeatures):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Print received feature names for debugging
        print("Feature names received from API input:", input_df.columns.tolist())

        # Ensure columns match the model's feature names (Force correct order)
        input_df = input_df[model.feature_names]

        # Convert DataFrame to XGBoost DMatrix
        dmatrix = xgb.DMatrix(input_df, feature_names=model.feature_names)

        # Make prediction
        prediction = model.predict(dmatrix)

        return {"predicted_price": float(prediction[0])}

    except Exception as e:
        return {"error": str(e)}

