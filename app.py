import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
from PIL import Image
import os

# Load the trained model
model = load_model('C:\\Users\\Yaswanth kumar Reddy\\Desktop\\Project\\model.h5')


# Define a function to preprocess the input image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize image to target size
    img_array = np.array(img) / 255.  # Convert image to array and normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

app = Flask(__name__)

# Define the home page
@app.route('/')
def home():
    return render_template('Index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['file']

    # Read the image file
    img = Image.open(file)

    # Preprocess the image
    img_array = preprocess_image(img)

    # Make a prediction
    prediction = model.predict(img_array)

    # Convert the prediction to a string
    if np.argmax(prediction) == 0:
        prediction_str = 'Over_Riped'
    elif np.argmax(prediction) == 1:
        prediction_str = 'Perfectly_Riped'
    else:
        prediction_str = 'Under_Riped'

    # Save the image file to a temporary location 
    file_path = 'temp.jpg'
    img.save(file_path)

    # Redirect to the result page with prediction and image data
    return redirect(url_for('result', prediction=prediction_str, file_path=file_path))

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    file_path = request.args.get('file_path')
    with open(file_path, 'rb') as f:
        img_base64 = base64.b64encode(f.read()).decode()
    return render_template('Result.html', prediction=prediction, image=img_base64)

if __name__ == '__main__':
    app.run(debug=True)
