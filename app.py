from flask import Flask, render_template, request
import pytesseract
from PIL import Image
import os

app = Flask(__name__)

# Path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\salia\OneDrive\Documents\FunWithPython\tesseract.exe"

# Simple fraud keywords (replace with ML model later if needed)
fraud_keywords = ["lottery", "prize", "winner", "urgent", "bank", "otp", "money"]

def check_fraud(text):
    for word in fraud_keywords:
        if word.lower() in text.lower():
            return "⚠️ Fraudulent Message Detected!"
    return "✅ Safe Message"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "screenshot" not in request.files:
        return "No file uploaded", 400
    
    file = request.files["screenshot"]
    if file.filename == "":
        return "No file selected", 400

    # Save the file temporarily
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # OCR extraction
    img = Image.open(filepath)
    extracted_text = pytesseract.image_to_string(img)

    # Check fraud
    result = check_fraud(extracted_text)

    return render_template("index.html", result=result, text=extracted_text, image=filepath)

if __name__ == "__main__":
    app.run(debug=True)
