from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imutils import paths
from matplotlib import pyplot as plt
import numpy as np
import argparse
import json
import cv2
import os


# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="../../dataset/train", help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="Skin_Model.model", help="path to output model")
args = vars(ap.parse_args())
# Initialize data and labels lists
data = []
labels = []
# Training parameters
INIT_LR = 1e-4
EPOCHS = 25
BS = 4  # batch size
# Load images and labels
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    data.append(image)
    labels.append(label)
# Convert to numpy arrays and scale pixel intensities to [0,1]
data = np.array(data) / 255.0
labels = np.array(labels)
# One-hot encode labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# Split data into train and test sets
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.20, random_state=42
)
# Data augmentation for training
trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")
# Load base VGG19 model without top layers
baseModel = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
# Build the head of the model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(5, activation="softmax")(headModel)  # 5 classes
# Combine base and head into final model
model = Model(inputs=baseModel.input, outputs=headModel)
# Freeze base model layers
for layer in baseModel.layers:
    layer.trainable = False
# Compile model with categorical crossentropy loss
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR)  # Removed deprecated decay argument
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# Train the model
print("[INFO] training head...")
H = model.fit(
    trainAug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS
)

# Evaluate the model
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
# Save the model
print("[INFO] saving skin cancer model...")
model.save(args["model"], save_format="h5")
# Classification report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))
# Confusion matrix and metrics
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = np.sum(cm)
# Overall accuracy for multiclass
acc = np.trace(cm) / total
print(cm)
print(f"Overall accuracy: {acc:.4f}")
# Per-class sensitivity and specificity
for i, class_name in enumerate(lb.classes_):
    tp = cm[i, i]
    fn = np.sum(cm[i, :]) - tp
    fp = np.sum(cm[:, i]) - tp
    tn = total - tp - fn - fp
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"Class '{class_name}': sensitivity={sensitivity:.4f}, specificity={specificity:.4f}")
# Plot training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history.get("val_loss", []), label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history.get("val_accuracy", []), label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# Optional: Fix for custom dataset class warning (if you have one)
# class PyDataset(tf.keras.utils.Sequence):
#     def __init__(self, *args, **kwargs):
#         super().__init__(**kwargs)
#         # your code here