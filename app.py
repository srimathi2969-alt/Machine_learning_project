from fastapi import FastAPI
import joblib
import pandas as pd 
app=FastAPI()

model=joblib.load("models/student_models.pkl")
@app.get("/")
def home():
    return{"message":"Student performance prediction api running"}
@app.post("/predict")

def predict(study_hours:float,attendance:float,previous_marks:float):
    input_data=pd.DataFrame([{"study_hour":study_hours,"attendance":attendance,"previous_mark":previous_marks}])
    prediction=model.predict(input_data)
    return {"prediction_result":prediction[0]}