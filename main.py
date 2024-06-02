import os
import sys
import numpy as np
import pandas as pd
import cv2
from flask import Flask, request, render_template_string, send_file, redirect, url_for

app = Flask(__name__)

# Update this path to the correct location of your haarcascade file
cascade_path = 'haarcascade_face.xml'
faceCascade = cv2.CascadeClassifier(cascade_path)

if faceCascade.empty():
    raise Exception(f"Error loading cascade file. Please ensure '{cascade_path}' is in the correct location.")

def create():
    if not "Training_Faces" in os.listdir("."):
        os.mkdir("Training_Faces")
    else:
        return
    
    label = 0
    i = 1
    arr = []

    for dirname, dirnames, filenames in os.walk('Training_Data'):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                    abs_path = os.path.join(subject_path, filename)
                    image = cv2.imread(abs_path)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=0)
                    for (x, y, w, h) in faces:
                        os.chdir("Training_Faces")
                        cv2.imwrite(f"{label}_{i}.jpg", gray[y-15:y+h+15, x-15:x+w+15])
                        arr.append([f"Training_Faces/{label}_{i}.jpg", label])
                        os.chdir("../")
                        i += 1
    np.savetxt('train_faces.csv', arr, delimiter=',', fmt='%s')
    print("CSV CREATED!")

def train():
    # Create 'train_faces.csv', which contains the images and their corresponding labels
    create()
    
    # Face Recognizer using Linear Binary Pattern Histogram Algorithm
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Read csv file using pandas
    data = pd.read_csv('train_faces.csv').values
    
    images = []
    labels = []
    
    for ix in range(data.shape[0]):
        img = cv2.imread(data[ix][0], cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img, (200, 200))  # Resize image to 200x200
        images.append(resized_img)
        labels.append(int(data[ix][1]))
    
    face_recognizer.train(images, np.array(labels))
    return face_recognizer

face_recog = train()

def test(test_img, face_recognizer):
    image = cv2.imread(test_img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=0)
    
    if len(faces) == 0:
        print("No faces detected.")
    
    for (x, y, w, h) in faces:
        sub_img = gray[y:y+h, x:x+w]
        resized_sub_img = cv2.resize(sub_img, (200, 200))  # Resize sub image to 200x200
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Predict label of detected face
        pred_label, confidence = face_recognizer.predict(resized_sub_img)
        
        cv2.putText(image, f"Label: {pred_label}, Confidence: {confidence}", (x, y-5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)
    
    output_path = 'static/recognized_image.jpg'
    cv2.imwrite(output_path, image)
    return output_path

# Flask Routes
@app.route('/')
def index():
    return render_template_string(
      '''
        <!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Face Recognition App</title>
    <style>
      body {
        font-family: Arial, sans-serif;
        background-image: url('https://png.pngtree.com/thumb_back/fh260/background/20210806/pngtree-border-black-black-gold-polygon-background-business-image_757234.jpg');
        background-size: cover;
        background-position: center;
        margin: 0;
        padding: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
      }
      .container {
        text-align: center;
        background-color: #FFFFFF;
        padding: 40px;
        border-radius: 12px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.2);
        animation: fadeIn 1s ease-in-out;
        width: 60%;
        max-width: 600px;
      }
      @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
      }
      h1 {
        margin-bottom: 30px;
        color: #333;
        font-size: 2.5em;
        border-bottom: 2px solid #ccc;
        padding-bottom: 10px;
      }
      p {
        font-size: 1.2em;
        color: #666;
        margin-bottom: 20px;
      }
      form {
        margin-bottom: 20px;
      }
      input[type="file"] {
        margin: 20px 0;
        font-size: 1em;
      }
      button {
        background-color: #258E28;
        color: black;
        padding: 15px 25px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s, transform 0.3s;
        font-size: 1.1em;
        margin: 5px;
      }
      button:hover {
        background-color: #2EF438;
        transform: scale(1.05);
      }
      .download-btn {
        background-color: #0091C1;
      }
      .download-btn:hover {
        background-color: #00AEFF;
        transform: scale(1.05);
      }
      .reset-btn {
        background-color: #C50D00;
      }
      .reset-btn:hover {
        background-color: #FF0000;
        transform: scale(1.05);
      }
      img {
        max-width: 100%;
        height: auto;
        margin-top: 30px;
        border-radius: 12px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
        animation: fadeIn 2s ease-in-out;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Face Recognition App</h1>
      <p>Upload an image to recognize faces</p>
      <form method="post" action="/upload" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <br>
        <button type="submit">Upload and Recognize</button>
        <button type="button" class="reset-btn" onclick="resetForm()">Reset</button>
      </form>
      {% if image_url %}
      <h2>Recognized Image</h2>
      <img src="{{ image_url }}" alt="Recognized Image">
      <br>
      <a href="{{ image_url }}" download="recognized_image.jpg">
        <button class="download-btn">Download</button>
      </a>
      {% endif %}
    </div>
    <script>
      function resetForm() {
        window.location.href = '/';
      }
    </script>
  </body>
</html>

    ''', image_url=request.args.get('image_url'))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)
        recognized_image_path = test(filepath, face_recog)
        return redirect(url_for('index', image_url=recognized_image_path))

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
