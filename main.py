from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = FastAPI()
model = load_model("deepfake_model.h5")

IMG_SIZE = 224

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body>
            <h2>Deepfake Detection</h2>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file"/>
                <input type="submit"/>
            </form>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        result = f"Fake ({prediction*100:.2f}%)"
    else:
        result = f"Real ({(1-prediction)*100:.2f}%)"

    return {"result": result}