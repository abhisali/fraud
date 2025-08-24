from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pytesseract
from PIL import Image
import os
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for React

# Path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Enhanced fraud keywords
strong_fraud_keywords = [
    "bank alert", "zelle payment", "fraud alert", "click here", "reply yes", 
    "urgent", "suspended", "verify now", "claim", "won", "lottery", "prize",
    "account blocked", "update details", "suspicious activity", "j.p. morgan",
    "chase bank", "free msg", "attempt a", "decline fraud alerts"
]

# Load the trained model
try:
    model = joblib.load("fraud_detector.pkl")
    use_ml_model = True
    print("✅ ML Model loaded successfully")
except Exception as e:
    use_ml_model = False
    print(f"⚠️ ML Model not found: {e}, using keyword matching")

def enhanced_fraud_check(text):
    """Enhanced and more precise fraud detection"""
    
    text_lower = text.lower()
    
    # POSITIVE indicators (safe message patterns)
    safe_patterns = [
        "say no to drugs", "anti narcotics", "crime branch", "police", 
        "health", "safety", "education", "community", "public service",
        "awareness", "helpline", "government", "ministry", "department",
        "traffic police", "fire safety", "blood donation", "vaccination",
        "election commission", "vote", "library", "school", "hospital"
    ]
    
    # Check for safe patterns first
    safe_matches = [pattern for pattern in safe_patterns if pattern in text_lower]
    if safe_matches and len(safe_matches) >= 1:
        return {
            "is_fraud": False,
            "confidence": 0.95,
            "method": "Safe Pattern Recognition",
            "matched_keywords": safe_matches,
            "reason": "Identified as legitimate public service/educational message"
        }
    
    # Strong fraud indicators (financial/urgent action required)
    strong_fraud_keywords = [
        "click here to claim", "reply yes", "account suspended", "verify immediately",
        "limited time offer", "act now", "claim prize", "won lottery", 
        "bank alert", "zelle payment", "fraud alert", "update details",
        "account blocked", "suspicious activity", "call immediately",
        "your account will be", "click to unlock", "verify your account"
    ]
    
    # Check for fraud patterns
    strong_matches = [kw for kw in strong_fraud_keywords if kw in text_lower]
    
    # Financial fraud patterns
    financial_fraud_patterns = [
        ("$" in text and "won" in text_lower),
        ("$" in text and "claim" in text_lower),
        ("$" in text and "prize" in text_lower),
        ("bank" in text_lower and "alert" in text_lower and "$" in text),
        ("reply yes" in text_lower and "$" in text),
        ("click" in text_lower and ("prize" in text_lower or "claim" in text_lower))
    ]
    
    # If multiple strong fraud indicators, likely fraud
    if len(strong_matches) >= 2 or any(financial_fraud_patterns):
        return {
            "is_fraud": True,
            "confidence": 0.9,
            "method": "Fraud Pattern Recognition",
            "matched_keywords": strong_matches,
            "reason": "Multiple fraud indicators detected"
        }
    
    # Use ML model with better error handling
    if use_ml_model:
        try:
            prediction = model.predict([text])[0]
            confidence = max(model.predict_proba([text])[0])
            
            # Don't override high-confidence safe predictions for legitimate messages
            if prediction == "safe" and confidence > 0.7:
                return {
                    "is_fraud": False,
                    "confidence": float(confidence),
                    "method": "Machine Learning Model",
                    "matched_keywords": safe_matches or [],
                    "prediction": prediction
                }
            
            # Don't override high-confidence fraud predictions
            if prediction == "fraud" and confidence > 0.8:
                return {
                    "is_fraud": True,
                    "confidence": float(confidence),
                    "method": "Machine Learning Model",
                    "matched_keywords": strong_matches,
                    "prediction": prediction
                }
            
            # For low-confidence predictions, use keyword analysis as backup
            if confidence < 0.7:
                keyword_result = keyword_fallback_analysis(text_lower, strong_matches, safe_matches)
                if keyword_result:
                    return keyword_result
            
            return {
                "is_fraud": prediction == "fraud",
                "confidence": float(confidence),
                "method": "Machine Learning Model",
                "matched_keywords": strong_matches or safe_matches,
                "prediction": prediction
            }
            
        except Exception as e:
            print(f"ML Model error: {e}")
    
    # Fallback keyword analysis
    return keyword_fallback_analysis(text_lower, strong_matches, safe_matches)

def keyword_fallback_analysis(text_lower, strong_matches, safe_matches):
    """Fallback analysis using keywords"""
    
    # If safe patterns found, mark as safe
    if safe_matches:
        return {
            "is_fraud": False,
            "confidence": 0.85,
            "method": "Keyword Analysis - Safe",
            "matched_keywords": safe_matches,
            "reason": "Contains legitimate service keywords"
        }
    
    # If fraud patterns found, mark as fraud
    if strong_matches:
        fraud_score = len(strong_matches)
        confidence = min(fraud_score * 0.4, 0.9)
        return {
            "is_fraud": True,
            "confidence": confidence,
            "method": "Keyword Analysis - Fraud",
            "matched_keywords": strong_matches,
            "fraud_score": fraud_score
        }
    
    # No clear indicators - default to safe with low confidence
    return {
        "is_fraud": False,
        "confidence": 0.6,
        "method": "Default Classification",
        "matched_keywords": [],
        "reason": "No clear fraud indicators found"
    }


def check_fraud(text):
    """Simple function for HTML template compatibility"""
    result = enhanced_fraud_check(text)
    return "⚠️ Fraudulent Message Detected!" if result["is_fraud"] else "✅ Safe Message"

def check_fraud_analysis(text):
    """Detailed analysis function for API"""
    return enhanced_fraud_check(text)

# Original route for HTML template
@app.route("/")
def index():
    return render_template("index.html")

# Original upload route for HTML template
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

    try:
        # OCR extraction
        img = Image.open(filepath)
        extracted_text = pytesseract.image_to_string(img)
        print(f"HTML Upload - Extracted text: {extracted_text}")

        # Check fraud using enhanced method
        result = check_fraud(extracted_text)
        print(f"HTML Upload - Result: {result}")

        return render_template("index.html", result=result, text=extracted_text, image=filepath)
    
    except Exception as e:
        print(f"Error in HTML upload: {e}")
        return f"Error processing image: {str(e)}", 500

# API endpoint for React
@app.route("/api/analyze", methods=["POST"])
def analyze_screenshot():
    try:
        print("📥 Received analyze request")
        
        if "screenshot" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["screenshot"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        print(f"📁 Processing file: {file.filename}")

        # Create uploads directory if it doesn't exist
        upload_folder = "uploads"
        os.makedirs(upload_folder, exist_ok=True)
        
        # Save file temporarily
        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)
        print(f"💾 File saved to: {filepath}")

        try:
            # OCR extraction
            print("🔍 Starting OCR extraction...")
            img = Image.open(filepath)
            extracted_text = pytesseract.image_to_string(img).strip()
            print(f"📝 API - Extracted text: {extracted_text}")
            
            if not extracted_text:
                print("⚠️ No text found in image")
                return jsonify({
                    "success": True,
                    "extracted_text": "",
                    "fraud_analysis": {
                        "is_fraud": False,
                        "confidence": 0,
                        "method": "No text detected",
                        "message": "No readable text found in image"
                    }
                })

            # Fraud detection using enhanced method
            print("🔍 Starting fraud analysis...")
            fraud_result = check_fraud_analysis(extracted_text)
            print(f"✅ API - Analysis result: {fraud_result}")

            return jsonify({
                "success": True,
                "extracted_text": extracted_text,
                "fraud_analysis": fraud_result
            })

        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"🗑️ Cleaned up file: {filepath}")

    except Exception as e:
        print(f"❌ Error in analyze_screenshot: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy", 
        "ml_model_loaded": use_ml_model,
        "tesseract_configured": True
    })

if __name__ == "__main__":
    # First, train the model if it doesn't exist
    if not os.path.exists("fraud_detector.pkl"):
        print("🔄 Training fraud detection model...")
        try:
            exec(open("fraud_model.py").read())
            print("✅ Model trained and saved")
        except Exception as e:
            print(f"❌ Failed to train model: {e}")
    
    print("🚀 Starting Flask server...")
    app.run(debug=True, port=5000, host='0.0.0.0')
