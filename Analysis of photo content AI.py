import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(img_array)
    return preprocessed_img

def predict_image_content(model, img_path):
    preprocessed_img = load_and_preprocess_image(img_path)
    predictions = model.predict(preprocessed_img)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    print("Predictions:")
    for _, label, confidence in decoded_predictions:
        print(f"{label}: {confidence:.2f}")

if __name__ == "__main__":
    # Load MobileNetV2 model only once
    model = MobileNetV2(weights='imagenet')

    # Replace "path/to/your/image.jpg" with the actual path to the image you want to test
    image_path = "king.jpeg"
    predict_image_content(model, image_path)
