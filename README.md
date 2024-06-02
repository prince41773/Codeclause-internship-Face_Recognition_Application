# Face Recognition App

A web application for face recognition using OpenCV and Flask, allowing users to upload images, detect, and recognize faces with ease.

## Features

- Upload an image to recognize faces
- Displays recognized faces with labels and confidence scores
- Download the recognized image with annotations

## Requirements

- Python 3.x
- Flask
- OpenCV
- NumPy
- pandas

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/prince41773/Codeclause-internship-Face_Recognition_Application.git
   cd Codeclause-internship-Face_Recognition_Application
2. Install the required packages:
   ```bash
   pip install -r requirements.txt

3. Download the Haar Cascade file for face detection and place it in the project directory. You can get it from here.

4. Ensure you have the im.jpg file in the project directory for the background image.

## Usage

1. Run the application:

   ```bash
   python main.py

2. Open your web browser and go to http://127.0.0.1:5000/.

3. Upload an image and view the recognized faces.

## Project Structure
- main.py: Main Flask application file
- haarcascade_face.xml: Haar Cascade file for face detection
- im.jpg: Background image for the web interface
- Training_Data/: Directory containing training images
- Training_Faces/: Directory where processed training images will be saved
- static/: Directory for saving output images
- templates/: Directory for HTML templates
