from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import PIL.Image

app = Flask(__name__)

# Load a pre-trained deep learning model for image classification
model = tf.keras.applications.InceptionV3(include_top=True, weights='imagenet')

# Define a function to preprocess the image
def preprocess(image):
    img = PIL.Image.open(image)
    img = img.resize((299, 299))
    img_array = np.array(img)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define a function to generate hashtags for an image
def generate_hashtags(image):
    img_array = preprocess(image)
    features = model.predict(img_array)
    hashtags = []
    for i in range(10):
        idx = np.argmax(features)
        word = tf.keras.applications.inception_v3.decode_predictions(features, top=10)[0][i][1]
        hashtags.append(word)
        features[0][idx] = -1
    return hashtags

# Define a route to handle file upload and hashtag generation
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({"error": "No file uploaded"}), 400
        image = request.files['file']
        hashtags = generate_hashtags(image)
        return jsonify(hashtags)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
