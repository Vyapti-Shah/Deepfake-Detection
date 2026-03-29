#run this in terminal: python train.py 

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 224

# Load dataset
def load_data(data_dir):
    X, y = [], []
    
    for label, folder in enumerate(["real", "fake"]):
        path = os.path.join(data_dir, folder)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            
            X.append(img)
            y.append(label)
    
    return np.array(X), np.array(y)

print("Loading dataset...")
X, y = load_data("dataset")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load ResNet50
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))

for layer in base_model.layers:
    layer.trainable = False

# Custom layers
x = Flatten()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=32
)

# Save model
model.save("deepfake_model.h5")

# EVALUATION
print("Evaluating model...")

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

os.makedirs("results", exist_ok=True)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")

# Accuracy Graph
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Plot")
plt.savefig("results/accuracy_plot.png")

print("Training complete. Results saved in /results folder.")