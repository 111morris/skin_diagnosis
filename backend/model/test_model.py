import numpy as np, json, os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Skin_Model.h5')
LABELS_PATH = os.path.join(os.path.dirname(__file__), 'labels.json')


model = load_model(MODEL_PATH)
classes = json.load(open(LABELS_PATH))


img = image.load_img('SAMPLE.jpg', target_size=(224, 224)) # put a real image here
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)


preds = model.predict(x)[0]
idx = int(np.argmax(preds))
print('Pred vector:', preds)
print('Class:', classes[idx], 'Confidence:', float(preds[idx]))