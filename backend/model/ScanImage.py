from tensorflow import keras
import cv2
import numpy as np 
import argparse 


# Construct the argument parser and parse the arguments 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="path to the image to test")
args = vars(ap.parse_args())

# load the model 
model = keras.models.load_model("Skin_Model.h5")

# Define labels 
labels = ["Acne", "Hair Fall", "Nail Fungus", "Normal", "Skin Allergy"]

# Preprocess the image 
image = cv2.imread(args["image"])

if image is None: 
 print("[E] Could not read the image. Please check the file path.")
 exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224)) # Normalize the image
""" image = np.array(image) / 255.0 # Add batch dimension
 """

image = np.expand_dims(image, axis=0)
image = np.array(image) / 255.0

# Make prediction 
predictions = model.predict(image)


# Get the maximum probability and corresponding label 
max_prob = np.max(predictions) * 100
predicted_label = labels[np.argmax(predictions)]


# Display the result
print(f"[D] {predicted_label}")
print(f"[P] {round(max_prob, 2)}%")