import cv2
import numpy as np
from collections import deque
import os

def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")
    print(f"Looking for model files in: {models_dir}")

    face_prototxt = os.path.join(models_dir, "deploy.prototxt")
    face_model = os.path.join(models_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    age_prototxt = os.path.join(models_dir, "age_deploy.prototxt")
    age_model = os.path.join(models_dir, "age_net.caffemodel")
    gender_prototxt = os.path.join(models_dir, "gender_deploy.prototxt")
    gender_model = os.path.join(models_dir, "gender_net.caffemodel")

    required_files = [
        (face_prototxt, "https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt"),
        (face_model, "https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"),
        (age_prototxt, "https://github.com/GilLevi/AgeGenderDeepLearning/blob/master/models/age_net_deploy.prototxt"),
        (age_model, "https://github.com/GilLevi/AgeGenderDeepLearning/blob/master/models/age_net.caffemodel"),
        (gender_prototxt, "https://github.com/GilLevi/AgeGenderDeepLearning/blob/master/models/gender_net_deploy.prototxt"),
        (gender_model, "https://github.com/GilLevi/AgeGenderDeepLearning/blob/master/models/gender_net.caffemodel")
    ]

    for file_path, url in required_files:
        if not os.path.isfile(file_path):
            print(f"Error: {file_path} not found.")
            print(f"Please download it from: {url}")
            print(f"and place it in the models directory: {models_dir}")
            raise FileNotFoundError(f"Required file not found: {file_path}")

    face_detector = cv2.dnn.readNet(face_prototxt, face_model)
    age_net = cv2.dnn.readNet(age_prototxt, age_model)
    gender_net = cv2.dnn.readNet(gender_prototxt, gender_model)
    return face_detector, age_net, gender_net

def detect_age_gender(frame, face_detector, age_net, gender_net, age_history):
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    face_detector.setInput(blob)
    detections = face_detector.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            
            face = frame[y:y1, x:x1]
            
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            
            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            age = int(np.argmax(age_preds) * 4.76 + 2)
            
            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender = "Male" if gender_preds[0][0] > gender_preds[0][1] else "Female"
            
            age_history.append(age)
            smoothed_age = int(sum(age_history) / len(age_history))
            
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            label = f"Age: {smoothed_age}, Gender: {gender}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

def main():
    try:
        face_detector, age_net, gender_net = load_models()
        age_history = deque(maxlen=10)
        
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = detect_age_gender(frame, face_detector, age_net, gender_net, age_history)
            
            cv2.imshow("Age and Gender Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please download the required model files and try again.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
