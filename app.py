from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
model = load_model("fish_disease_model.h5")

# Replace with your actual class names if needed
class_names = sorted(os.listdir("Train")) if os.path.exists("Train") else ["Columnaris", "Fin Rot", "Ich"]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    return jsonify({'prediction': predicted_class})
