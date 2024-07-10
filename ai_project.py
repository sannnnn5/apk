import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Load the pre-trained model
model = MobileNetV2(weights='imagenet')

# Define a comprehensive set of animal-related classes
animal_classes = {
    'dog', 'cat', 'lion', 'tiger', 'bear', 'elephant', 'monkey', 'horse', 'sheep', 'cow', 'goat', 'deer', 'bird', 'fish',
    'reptile', 'insect', 'snake', 'spider', 'frog', 'turtle', 'hamster', 'guinea_pig', 'rabbit', 'chicken', 'rooster',
    'penguin', 'dolphin', 'whale', 'shark', 'octopus', 'crab', 'lobster', 'bee', 'butterfly', 'ant', 'bat', 'buffalo',
    'camel', 'cheetah', 'chimpanzee', 'crocodile', 'donkey', 'eagle', 'flamingo', 'giraffe', 'goose', 'hawk', 'hyena',
    'jaguar', 'kangaroo', 'koala', 'leopard', 'lizard', 'lynx', 'mole', 'moose', 'ostrich', 'otter', 'owl', 'panda',
    'parrot', 'peacock', 'pelican', 'pigeon', 'platypus', 'polar_bear', 'porcupine', 'raccoon', 'rat', 'raven', 'rhinoceros',
    'scorpion', 'seal', 'seahorse', 'skunk', 'sloth', 'snail', 'squid', 'squirrel', 'swan', 'vulture', 'walrus', 'wombat',
    'woodpecker', 'yak', 'zebra'
}

# Define a confidence threshold
confidence_threshold = 0.5

def load_and_preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)
    except Exception as e:
        print(f"Error loading or preprocessing image: {e}")
        return None

def classify_image(img_path):
    img_array = load_and_preprocess_image(img_path)
    if img_array is not None:
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    else:
        return None

def annotate_image(img_path, predictions, label):
    try:
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        
        font = ImageFont.truetype("arial.ttf", 20)
        
        
        lines = [f"This image contains a {label}"]
        for i, prediction in enumerate(predictions):
            lines.append(f"Prediction {i+1}: {prediction[1]} (confidence: {prediction[2]*100:.2f}%)")
        
        
        total_text = "\n".join(lines)
        text_bbox = draw.textbbox((0, 0), total_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        width, height = img.size
        x = (width - text_width) / 2
        y = height - text_height - 10
        
        
        draw.rectangle([x, y, x + text_width, y + text_height], fill="white")
        
       
        draw.text((x, y), total_text, font=font, fill="red")
        return img
    except Exception as e:
        print(f"Error annotating image: {e}")
        return None

def main(img_path):
    predictions = classify_image(img_path)
    if predictions:
        for i, prediction in enumerate(predictions):
            print(f"Prediction {i+1}: {prediction[1]} (confidence: {prediction[2]*100:.2f}%)")
        
        
        high_confidence_animal = any(pred[1].lower() in animal_classes and pred[2] > confidence_threshold for pred in predictions)
        
        if high_confidence_animal:
            label = "animal"
        else:
            label = "person"
        
       
        annotated_img = annotate_image(img_path, predictions, label)
        if annotated_img:
            plt.imshow(annotated_img)
            plt.axis('off')
            plt.show()
        else:
            print("Failed to annotate the image.")
    else:
        print("Failed to classify the image.")

if __name__ == "__main__":
    

    img_path = 'C:\\Users\\user\\Desktop\\ai2\\Untitled.jpg' 
    main(img_path)

