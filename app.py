# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import os
# from PIL import Image
# import traceback

# app = Flask(__name__)

# # Allow specific origin (Vite frontend)
# CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}})

# # Load the trained model
# MODEL_PATH = 'crop_disease_model.h5'
# try:
#     print("Loading model...")
#     model = load_model(MODEL_PATH)
#     print("Model loaded successfully!")
#     print(model.summary())
#     print(f"Model input shape: {model.input_shape}")
# except Exception as e:
#     print(f"Error loading model: {str(e)}")
#     exit(1)

# # Class labels (ensure order matches your model's output)
# CLASS_NAMES = [
#     "Cashew healthy", "Cashew red rust", "Cassava brown spot", "Cassava healthy",
#     "Guava scab", "Maize healthy", "Maize leaf blight", "Maize streak virus",
#     "Pumpkin healthy", "Pumpkin powdery mildew", "Tomato healthy", "Tomato leaf curl",
#     "Tomato septoria"
# ]

# def preprocess_image(file_path):
#     try:
#         img = Image.open(file_path)

#         # Dynamically adjust image size
#         input_shape = model.input_shape
#         if len(input_shape) == 4:
#             IMG_SIZE = (input_shape[1], input_shape[2])
#             print(f"Adjusting image size to: {IMG_SIZE}")
#             img = img.resize(IMG_SIZE)

#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0) / 255.0

#         return img_array

#     except Exception as e:
#         print(f"Error during image preprocessing: {str(e)}")
#         raise e

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         print("No file uploaded")
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['file']

#     if file.filename == '':
#         print("Empty filename")
#         return jsonify({"error": "Empty filename"}), 400

#     # Save and preprocess image
#     file_path = "temp_image.jpg"
#     file.save(file_path)

#     try:
#         print("Processing image...")
#         img_array = preprocess_image(file_path)

#         print("Making predictions...")
#         predictions = model.predict(img_array)
#         predicted_class = CLASS_NAMES[np.argmax(predictions)]
#         confidence = float(np.max(predictions))

#         print(f"Prediction: {predicted_class}, Confidence: {confidence}")

#     except Exception as e:
#         print("Prediction error:", traceback.format_exc())
#         return jsonify({"error": f"Prediction error: {str(e)}"}), 500

#     finally:
#         if os.path.exists(file_path):
#             os.remove(file_path)

#     return jsonify({"prediction": predicted_class, "confidence": confidence})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from PIL import Image
import traceback

app = Flask(__name__)

# Allow specific origin (Vite frontend)
CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}})

# Load the trained model
MODEL_PATH = 'crop_disease_model.h5'
try:
    print("Loading model...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
    print(model.summary())
    print(f"Model input shape: {model.input_shape}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit(1)

# Class labels (ensure order matches your model's output)
CLASS_NAMES = [
    "Cashew healthy", "Cashew red rust", "Cassava brown spot", "Cassava healthy",
    "Guava scab", "Maize healthy", "Maize leaf blight", "Maize streak virus",
    "Pumpkin healthy", "Pumpkin powdery mildew", "Tomato healthy", "Tomato leaf curl",
    "Tomato septoria"
]

def preprocess_image(file_path):
    try:
        img = Image.open(file_path)

        # Dynamically adjust image size
        input_shape = model.input_shape
        if len(input_shape) == 4:
            IMG_SIZE = (input_shape[1], input_shape[2])
            print(f"Adjusting image size to: {IMG_SIZE}")
            img = img.resize(IMG_SIZE)

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        return img_array

    except Exception as e:
        print(f"Error during image preprocessing: {str(e)}")
        raise e

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("No file uploaded")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '':
        print("Empty filename")
        return jsonify({"error": "Empty filename"}), 400

    # Save and preprocess image
    file_path = "temp_image.jpg"
    file.save(file_path)

    try:
        print("Processing image...")
        img_array = preprocess_image(file_path)

        print("Making predictions...")
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        print(f"Prediction: {predicted_class}, Confidence: {confidence}")

    except Exception as e:
        print("Prediction error:", traceback.format_exc())
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify({"prediction": predicted_class, "confidence": confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)