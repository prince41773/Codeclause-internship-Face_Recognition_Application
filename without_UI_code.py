import os
import sys
import numpy as np
import pandas as pd
import cv2

if len(sys.argv) < 2:
    print("Please add Test Image Path")
    sys.exit()

test_img = sys.argv[1]

# Update this path to the correct location of your haarcascade file
cascade_path = 'haarcascade_face.xml'
faceCascade = cv2.CascadeClassifier(cascade_path)

if faceCascade.empty():
    print(f"Error loading cascade file. Please ensure '{cascade_path}' is in the correct location.")
    sys.exit()

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
    cv2.imshow('Face Recognition', image)
    # Press Esc to Close Window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    face_recog = train()
    test(test_img, face_recog)