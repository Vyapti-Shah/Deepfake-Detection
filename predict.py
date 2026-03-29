import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 224

model = load_model("model/deepfake_model.h5")

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        return "Fake", float(prediction)
    else:
        return "Real", float(1 - prediction)