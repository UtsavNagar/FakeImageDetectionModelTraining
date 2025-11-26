import os
import cv2
import shutil
import mediapipe as mp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

# 1. PATH SETUP
ROOT_DIR = "/home/utsav/Desktop/FakeImageDetectionModelTraining"
RAW_DATASET_DIR = os.path.join(ROOT_DIR, "")
MERGED_DIR = os.path.join(ROOT_DIR, "dataset")
CROPPED_DIR = os.path.join(ROOT_DIR, "dataset_cropped")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(MERGED_DIR + "/real", exist_ok=True)
os.makedirs(MERGED_DIR + "/fake", exist_ok=True)
os.makedirs(CROPPED_DIR + "/real", exist_ok=True)
os.makedirs(CROPPED_DIR + "/fake", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. MERGE DATASET (REAL + FAKE FROM BOTH SOURCES)
SOURCE_REAL = [
    f"{ROOT_DIR}/real_and_fake_face/training_real",
    f"{ROOT_DIR}/real_and_fake_face_detection/real_and_fake_face/training_real"
]

SOURCE_FAKE = [
    f"{ROOT_DIR}/real_and_fake_face/training_fake",
    f"{ROOT_DIR}/real_and_fake_face_detection/real_and_fake_face/training_fake"
]

def merge_dataset():
    print("Merging dataset...")

    # Copy REAL
    for src in SOURCE_REAL:
        for img in os.listdir(src):
            shutil.copy(os.path.join(src, img), os.path.join(MERGED_DIR, "real"))

    # Copy FAKE
    for src in SOURCE_FAKE:
        for img in os.listdir(src):
            shutil.copy(os.path.join(src, img), os.path.join(MERGED_DIR, "fake"))

    print("Dataset merged successfully!")

merge_dataset()

# 3. FACE CROPPING USING MEDIAPIPE

print("Starting face cropping...")

mp_face = mp.solutions.face_detection

def crop_face(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        return

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        if not results.detections:
            return

        bbox = results.detections[0].location_data.relative_bounding_box
        h, w, _ = img.shape
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        w = int(bbox.width * w)
        h = int(bbox.height * h)

        face = img[y:y+h, x:x+w]

        if face.size > 0:
            cv2.imwrite(save_path, face)

# Apply cropping
for cls in ["real", "fake"]:
    src_folder = os.path.join(MERGED_DIR, cls)
    dst_folder = os.path.join(CROPPED_DIR, cls)

    for img in os.listdir(src_folder):
        crop_face(os.path.join(src_folder, img), os.path.join(dst_folder, img))

print("Face cropping completed!")

# 4. TRAINING PREP
img_size = 224
batch_size = 8

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)

train_ds = datagen.flow_from_directory(
    CROPPED_DIR,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
    subset="training"
)

val_ds = datagen.flow_from_directory(
    CROPPED_DIR,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
    subset="validation"
)

# 5. MODEL (MobileNetV3 Small Fine-Tuned)
base = MobileNetV3Small(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

base.trainable = True

inputs = layers.Input((224,224,3))
x = base(inputs)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 6. TRAIN MODEL
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# 7. EVALUATIONS
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.legend()
plt.title("Accuracy Curve")
plt.show()

plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.legend()
plt.title("Loss Curve")
plt.show()

y_pred = (model.predict(val_ds) > 0.5).astype("int32")
y_true = val_ds.classes

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["REAL", "FAKE"])
disp.plot(cmap="Blues")
plt.show()

print(classification_report(y_true, y_pred, target_names=["REAL", "FAKE"]))
print("Final Validation Accuracy:", accuracy_score(y_true, y_pred) * 100)

# 8. SAVE MODELS
model.save(os.path.join(MODEL_DIR, "deepfake_model.h5"))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(os.path.join(MODEL_DIR, "deepfake_model.tflite"), "wb") as f:
    f.write(tflite_model)

print("Model saved successfully!")
