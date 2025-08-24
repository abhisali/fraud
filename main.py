import joblib
import pytesseract
import cv2

# Path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\salia\OneDrive\Documents\FunWithPython\tesseract.exe"

# Load trained model
model = joblib.load("fraud_detector.pkl")

def extract_text(image_path):
    img = cv2.imread(image_path)
    text = pytesseract.image_to_string(img)
    return text.strip()

def predict_message(text):
    prediction = model.predict([text])[0]
    return prediction

if __name__ == "__main__":
    image_path = "screenshot.jpg"  # replace with your screenshot
    extracted_text = extract_text(image_path)
    print("Extracted Text:", extracted_text)

    result = predict_message(extracted_text)
    print("Prediction:", "🚨 FRAUD" if result == "fraud" else "✅ SAFE")
