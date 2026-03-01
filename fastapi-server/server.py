from fastapi import FastAPI, File, UploadFile
from model_helper import predict

app = FastAPI(title="Damage Prediction API")


@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):
    try:
        # Read uploaded file
        image_bytes = await file.read()

        # Save temporarily
        image_path = "temp_file.jpg"
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        # Get prediction
        prediction = predict(image_path)

        return {"prediction": prediction}

    except Exception as e:
        return {"error": str(e)}