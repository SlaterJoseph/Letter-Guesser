import base64
from io import BytesIO
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.python.keras.saving.save import load_model
from PIL import Image

app = Flask(__name__)
model = load_model('model/letter_predicter.keras')

def preprocess_image(raw_data) -> np.ndarray:
    img = Image.open(BytesIO(base64.b64decode(raw_data.split(',')[1])))
    img = img.convert('L')
    img = img.resize((28, 28))
    img = np.array(img).reshape(-1, 28, 28, 1) / 255.0
    return img

@app.route('/predict', methods=['POST'])
def predict() -> None:
    image_data = request.json['image']
    processed_image = preprocess_image(image_data)
    prediction = model.predict(processed_image)
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    result = {letter: float(prob) for letter, prob in zip(letters, prediction[0])}
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)