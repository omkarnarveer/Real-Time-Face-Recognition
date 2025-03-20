import os
import face_recognition
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# Define dataset and model paths
dataset_path = "dataset/"
model_path = "model/model.pkl"

# Distance threshold for classification
FACE_DISTANCE_THRESHOLD = 0.6

def load_and_split_data():
    """Loads images, extracts encodings, and splits into train/test sets."""
    all_images = []
    all_labels = []

    # Iterate through each person's folder
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):  
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                try:
                    # Load image and extract face encodings
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)

                    if encodings:
                        all_images.append(encodings[0])  # Store first encoding
                        all_labels.append(person_name)
                    else:
                        print(f"No face found in {image_path}")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    # Split data
    return train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

def train_model():
    """Trains the model and saves it to a pickle file."""
    train_images, test_images, train_labels, test_labels = load_and_split_data()

    # Save model
    model_data = {
        'train_images': train_images,
        'train_labels': train_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }
    os.makedirs("model", exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print("Model trained and saved successfully!")

    # Evaluate the model
    evaluate_model(test_images, test_labels, train_images, train_labels)

def evaluate_model(test_images, test_labels, train_images, train_labels):
    """Evaluates the model's accuracy on the test set."""
    correct_predictions = 0
    total_predictions = len(test_images)

    for i, test_encoding in enumerate(test_images):
        distances = face_recognition.face_distance(train_images, test_encoding)
        min_distance = min(distances)

        predicted_label = train_labels[np.argmin(distances)] if min_distance <= FACE_DISTANCE_THRESHOLD else "Unknown"
        
        if predicted_label == test_labels[i]:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions * 100
    print(f"Model Accuracy on Test Set: {accuracy:.2f}%")

if __name__ == '__main__':
    train_model()
