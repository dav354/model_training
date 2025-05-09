import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import mediapipe as mp
import subprocess
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# === Config ===
labels = {'none': 0, 'rock': 1, 'paper': 2, 'scissors': 3}
dataset_dir = "/app/training_data"
model_name = "/app/build/model.tflite"

print("[INFO] Extracting and augmenting hand landmarks...")

mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)
X, y = [], []

def add_noise(landmarks, noise_level=0.01):
    return landmarks + np.random.normal(0, noise_level, landmarks.shape)

def jitter_landmarks(landmarks, level=0.01):
    return landmarks + np.random.uniform(-level, level, landmarks.shape)

def rotate_landmarks(landmarks, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return landmarks @ R

def align_landmarks(landmarks):
    p0 = landmarks[0]
    p9 = landmarks[9]
    delta = p9 - p0
    angle = np.arctan2(delta[1], delta[0])
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s], [s, c]])
    return (landmarks - p0) @ R

total_images = 0
valid_samples = 0

for gesture in labels:
    gesture_dir = os.path.join(dataset_dir, gesture)
    for img_name in os.listdir(gesture_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        total_images += 1
        img_path = os.path.join(gesture_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for flip in [1, -1]:
            if flip == -1:
                flipped = cv2.flip(img_rgb, 1)
                results = mp_hands.process(flipped)
                if not results.multi_hand_landmarks:
                    continue
                landmarks = results.multi_hand_landmarks[0]
                coords = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
                coords[:, 0] = 1.0 - coords[:, 0]
            else:
                results = mp_hands.process(img_rgb)
                if not results.multi_hand_landmarks:
                    continue
                landmarks = results.multi_hand_landmarks[0]
                coords = np.array([[lm.x, lm.y] for lm in landmarks.landmark])

            coords = align_landmarks(coords)
            coords = coords - coords.min(axis=0)
            coords = coords / coords.max() if coords.max() > 0 else coords

            for angle in [0, 90, 180, 270, np.random.uniform(10, 350)]:
                rotated = rotate_landmarks(coords, angle)
                scaled = rotated * np.random.uniform(0.9, 1.1)
                offset = np.random.normal(0, 0.01, scaled.shape)
                jittered = jitter_landmarks(scaled + offset)
                flat_coords = add_noise(jittered.flatten())
                X.append(flat_coords)
                y.append(labels[gesture])
                valid_samples += 1

print(f"[INFO] Processed {total_images} images -> {valid_samples} augmented samples.")
X, y = np.array(X), np.array(y)
np.savez("/app/build/rps_landmarks.npz", X=X, y=y)

# === Step 2: Train model ===
print("[INFO] Training model...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train
)
class_weights = dict(enumerate(class_weights))

initial_lr = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=1000,
    decay_rate=0.95,
    staircase=True
)

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=4)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=1))
    return loss

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(42,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=focal_loss(),
    metrics=["accuracy"]
)

early_stop = EarlyStopping(monitor='val_loss', patience=15, min_delta=0.002, restore_best_weights=True)
model_ckpt = ModelCheckpoint("saved_rps_model", save_best_only=True, save_format="tf")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop, model_ckpt],
    verbose=1
)

# === Step 3: Evaluate performance ===
val_acc = history.history["val_accuracy"][-1]
print(f"[\U0001F4CA] Final Validation Accuracy: {val_acc*100:.2f}%")
baseline = 0.90
if val_acc > baseline:
    print(f"[✅] Model improved compared to baseline ({baseline*100:.1f}%)!")
else:
    print(f"[⚠️] Model underperformed compared to baseline ({baseline*100:.1f}%).")

# === Step 4: Convert to INT8 TFLite ===
print("[INFO] Converting to TFLite INT8...")
converter = tf.lite.TFLiteConverter.from_saved_model("saved_rps_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def rep_dataset():
    for i in range(min(300, len(X_train))):
        yield [X_train[i:i+1].astype(np.float32)]

converter.representative_dataset = rep_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

try:
    tflite_model = converter.convert()
    with open(model_name, "wb") as f:
        f.write(tflite_model)
    print(f"[✅] TFLite model saved to {model_name}")

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

except Exception as e:
    print("[❌] Error during TFLite conversion:", e)
