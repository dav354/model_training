import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import mediapipe as mp
import subprocess

# === Config ===
labels = {'none': 0, 'rock': 1, 'paper': 2, 'scissors': 3}
dataset_dir = "/app/training_data"
model_name = "/app/build/model.tflite" 

# === Step 1: Extract landmarks ===
print("[INFO] Extracting hand landmarks...")
mp_hands = mp.solutions.hands.Hands(static_image_mode=True)
X, y = [], []

for gesture in labels:
    gesture_dir = os.path.join(dataset_dir, gesture)
    for img_name in os.listdir(gesture_dir):
        img = cv2.imread(os.path.join(gesture_dir, img_name))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(img_rgb)
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            coords = np.array([[lm.x, lm.y] for lm in landmarks.landmark]).flatten()
            X.append(coords)
            y.append(labels[gesture])

X, y = np.array(X), np.array(y)
print(f"[INFO] Extracted {len(X)} samples.")
np.savez("/app/build/rps_landmarks.npz", X=X, y=y)

# === Step 2: Train model ===
print("[INFO] Training model...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(42,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(4, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

model.save("saved_rps_model")

# === Step 3: Convert to INT8 TFLite ===
print("[INFO] Converting to TFLite INT8...")
converter = tf.lite.TFLiteConverter.from_saved_model("saved_rps_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def rep_dataset():
    for i in range(min(100, len(X_train))):
        yield [X_train[i:i+1].astype(np.float32)]

converter.representative_dataset = rep_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()
with open(model_name, "wb") as f:
    f.write(tflite_model)

print(f"[✅] TFLite model saved to {model_name}")

# === Step 4: Compile for Edge TPU ===
print("[INFO] Compiling for Edge TPU...")
compile_result = subprocess.run(
    ["edgetpu_compiler", model_name, "-o", "/app/build"],
    capture_output=True,
    text=True
)

if compile_result.returncode == 0:
    print("[✅] Edge TPU model compiled successfully.")
    print(compile_result.stdout)
else:
    print("[❌] Error compiling model:")
    print(compile_result.stderr)
