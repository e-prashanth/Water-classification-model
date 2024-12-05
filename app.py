from flask import Flask, render_template, request, jsonify, redirect
import io
from flask_cors import CORS
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
CORS(app)
model = load_model(r"models/final.h5")

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/redirect_metrics')
def redirect_metrics():
    return redirect('/show_metrics')

@app.route("/show_metrics")
def metrics():
    return render_template("show_metrics.html")


@app.route("/classify", methods=["POST"])
def classify_image():
    uploaded_file = request.files["file"]

    if uploaded_file:
        image_data = uploaded_file.read()
        img = image.load_img(io.BytesIO(image_data), target_size=(128, 128))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = tf.image.rgb_to_grayscale(img) / 255.0

        result = model.predict(img)
        probability_ad, probability_ci, probability_cn = result[0]
        classes = ["water flood",'water logging','waste water']
        classification = classes[np.argmax(result[0])]

        accuracy, precision, recall, f1_score = (
            0.9778353483016695,
            0.9780466251883113,
            0.9778353483016695,
            0.9777159203982722,
        )  

        response_data = {
    "classification": classification,
    "probability_ad": float(probability_ad),  
    "probability_ci": float(probability_ci), 
    "probability_cn": float(probability_cn),  
    "accuracy": float(accuracy),  
    "precision": float(precision), 
    "recall": float(recall),
    "f1_score": float(f1_score)
    }

        return jsonify(response_data)
    return jsonify({"error": "No file provided"})

if __name__ == "__main__":
    app.run()