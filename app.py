
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import traceback

app = FastAPI()

# Load trained model
model = joblib.load("models/student_models.pkl")

class StudentData(BaseModel):
    study_hours: float
    attendance: float
    previous_marks: float

@app.get("/")
def home():
    return {"message": "Student Performance Prediction API "}

@app.post("/predict")
def predict(student: StudentData):
    try:
        input_data = pd.DataFrame([{
            "study_hours": student.study_hours,
            "attendance": student.attendance,
            "previous_marks": student.previous_marks
        }])

        print("Input data:")
        print(input_data)

        print("Model expected features:",
            getattr(model, "feature_names_in_", "Not available"))

        prediction = model.predict(input_data)

        result = prediction[0]

        # Convert NumPy values into normal Python values
        if hasattr(result, "item"):
            result = result.item()

        print("Prediction:", result)

        return {"prediction_result": result}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
