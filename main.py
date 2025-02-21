from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")

# Create FastAPI instance
app = FastAPI()

# Define the input payload structure
class InputPayload(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the Iris Flower Prediction API!"}

# Check if the model is loaded
@app.get("/model_status/")
def check_model():
    return {"message": "Model successfully loaded!"}

# Prediction endpoint
@app.post("/predict/")
def predict_species(input_data: InputPayload):
    try:
        # Convert input data to a NumPy array
        input_features = np.array([[input_data.sepal_length, input_data.sepal_width,
                                    input_data.petal_length, input_data.petal_width]])

        # Make a prediction
        prediction = model.predict(input_features)

        # Convert the NumPy output to a Python type
        predicted_species = str(prediction[0])  # Convert NumPy output to string

        return {"predicted_species": predicted_species}

    except Exception as e:
        return {"error": str(e)}

