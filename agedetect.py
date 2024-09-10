import cv2
import numpy as np
from deepface import DeepFace
from collections import deque

def detect_age(frame, face_cascade, age_history):
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        try:
            result = DeepFace.analyze(face, actions=['age'], enforce_detection=False)
            age = result[0]['age']
            
            # Update age history and calculate moving average
            age_history.append(age)
            smoothed_age = int(sum(age_history) / len(age_history))
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {smoothed_age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error in age detection: {e}")
    
    return frame

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize age history for smoothing
    age_history = deque(maxlen=10)  # Keeps track of last 10 age predictions
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = detect_age(frame, face_cascade, age_history)
        
        cv2.imshow("Age Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
