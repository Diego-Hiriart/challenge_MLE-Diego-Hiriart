import fastapi
from challenge.model import DelayModel
import pandas as pd

app = fastapi.FastAPI()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

# Method for data validation
def dataValidation(data: pd.DataFrame) -> bool:
    validData = True # If turned into false, data is invalid
    # Get valid arilines from data.csv
    airlines = pd.read_csv('../data/data.csv', skipinitialspace=True, usecols=["OPERA"])
    for index, row in data.iterrows():
        # Check for invalid month
        if row["MES"] <1 or row["MES"] >12:
            validData = False
            print("month")
            break
        # Check for invalid TIPOVUELO
        if row["TIPOVUELO"] != "N" and row["TIPOVUELO"] != "I":
            validData = False
            print("type")
            break
        # Check for invalid airline
        if row["OPERA"] not in airlines["OPERA"].values:
            validData = False
            print("airline")
            break
    return validData

@app.post("/predict", status_code=200)
async def post_predict(data: dict) -> dict:
    delayModel = DelayModel()
    features = pd.DataFrame.from_dict(data["flights"])
    # Validate data, return 400 if invalid
    if dataValidation(features) == False:
        raise fastapi.HTTPException(status_code=400, detail="invalid data")
    features = delayModel.preprocess(features)
    predictions = {"predict": delayModel.predict(features)}
    return predictions