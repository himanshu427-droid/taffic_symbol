import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Constants
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43

# Load the model
model = load_model("__model.h5")  

# GTSRB class labels (0–42)
class_labels = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

def predict_traffic_sign(image_path):
    # Load image (.ppm works fine)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Resize to 30x30 as used in training
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

    # Normalize pixel values to [0,1]
    image = image.astype("float32") / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)  # Shape: (1, 30, 30, 3)

    # Predict using the model
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Get label
    label = class_labels.get(predicted_class, f"Unknown class {predicted_class}")
    return predicted_class, label, confidence


if __name__ == "__main__":
    image_path = "gtsrb/28/00001_00005.ppm"  
    class_id, class_name, conf = predict_traffic_sign(image_path)
    print(f"Predicted Class ID: {class_id}")
    print(f"Traffic Sign: {class_name}")
    print(f"Confidence: {conf:.2f}")

    image_show = cv2.imread(image_path)
    image_show = cv2.resize(image_show, (IMG_WIDTH * 3, IMG_HEIGHT * 3))
    cv2.putText(image_show, class_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imshow("Prediction", image_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
