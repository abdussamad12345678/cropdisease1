from PIL import Image
import numpy as np

def predict_image(image):
    img = np.array(image)

    avg = img.mean()

    if avg < 100:
        return "Disease Detected"
    else:
        return "Healthy"
