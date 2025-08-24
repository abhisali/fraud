import pytesseract
from PIL import Image
import cv2
import os

# Path to Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\salia\OneDrive\Documents\FunWithPython\tesseract.exe"

def extract_text(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise TypeError("Error loading image. Check the format or path.")
    
    text = pytesseract.image_to_string(img)
    return text

if __name__ == "__main__":
    text = extract_text("screenshot.jpg")  # give full path if needed
    print("Extracted Text:", text)