import os
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from shutil import move

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make prediction
    preds = model.predict(x)
    return decode_predictions(preds, top=1)[0][0]

def organize_images(source_dir, dest_dir, confidence_threshold=0.7):
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Iterate through all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            file_path = os.path.join(source_dir, filename)
            try:
                # Predict the image category
                prediction = predict_image(file_path)
                category, confidence = prediction[1], prediction[2]

                # Create category folder if it doesn't exist
                category_path = os.path.join(dest_dir, category)
                if not os.path.exists(category_path):
                    os.makedirs(category_path)

                # Move the image to the appropriate category folder
                dest_path = os.path.join(category_path, filename)
                move(file_path, dest_path)
                print(f"Moved {filename} to {category} (confidence: {confidence:.2f})")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# Usage
source_directory = "source/images"
destination_directory = "organized/images"

organize_images(source_directory, destination_directory)
