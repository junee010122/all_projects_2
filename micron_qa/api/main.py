from fastapi import FastAPI, UploadFile, File
from api.schemas import TabularInput
from api.inference import predict_tabular, predict_image

app = FastAPI(title="Micron QA Inference API")

@app.post("/predict/tabular")
def predict_tab(data: TabularInput):
    pred = predict_tabular(data.features)
    return {"prediction": pred}

@app.post("/predict/image")
def predict_img(file: UploadFile = File(...)):
    pred = predict_image(file.file)
    return {"prediction": pred}

