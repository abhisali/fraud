import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

# Much more comprehensive and balanced dataset
data = {
    "message": [
        # FRAUD messages - Financial scams
        "You have won $1000 lottery, click here to claim",
        "Update your bank details immediately or account blocked", 
        "Congratulations! You won a free vacation",
        "Free Msg-J.P. Morgan Chase Bank Alert-Did You Attempt A Zelle Payment For The Amount of $5000.00? Reply YES or NO Or 1 To Decline Fraud Alerts",
        "URGENT: Your account has been suspended. Click to verify immediately",
        "You've won $50000! Claim your prize now by calling this number",
        "Bank of America: Suspicious activity detected. Verify your account now",
        "PayPal: Your account will be limited. Update payment method immediately",
        "Amazon: Your order of $999 has been placed. Cancel if not you by clicking here",
        "IRS: You owe back taxes. Pay immediately to avoid arrest",
        "Your credit card has been charged $500. Dispute now by clicking link",
        "Chase Bank: Fraud alert on your account. Reply YES to confirm transaction",
        "Wells Fargo: Your account is locked. Click to unlock immediately",
        "Apple ID suspended due to security issues. Verify now or lose access",
        "Microsoft: Your computer is infected. Call tech support immediately",
        "You are pre-approved for $10000 loan. Apply now, limited time offer",
        "Government grant available. Claim your $5000 now before it expires",
        "Walmart gift card winner! Claim $500 card now by calling",
        "Your package is held at customs. Pay $5 shipping fee to release",
        "Final notice: Your subscription will be charged $99 unless you cancel",
        "Bitcoin investment opportunity! Double your money in 24 hours",
        "Nigerian prince needs help transferring millions. Share in profits",
        "Tax refund of $2500 waiting. Click to claim before deadline",
        "Your Social Security number has been suspended. Call immediately",
        "Free iPhone 15! You're our 1000th visitor. Claim now!",
        
        # SAFE messages - Legitimate communications
        "Meeting at 10 AM tomorrow, please join",
        "Your package will be delivered today between 2-4 PM",
        "Project deadline is extended to next week",
        "Happy birthday! Hope you have a great day",
        "Lunch is ready, come downstairs",
        "Can you pick up milk on your way home?",
        "The weather is nice today, perfect for a walk",
        "Don't forget about the dentist appointment at 3 PM",
        "Movie starts at 8 PM, see you there",
        "Thanks for helping me with the project",
        "Good morning! Have a great day at work",
        "The meeting has been moved to conference room B",
        "Your flight is scheduled to depart at 2:30 PM",
        "Reminder: Parent-teacher conference tomorrow",
        "The restaurant reservation is confirmed for 7 PM",
        "Your library books are due next week",
        "Traffic is heavy on main street, take alternate route",
        "The gym class is cancelled today",
        "Your prescription is ready for pickup at the pharmacy",
        "School is closed due to weather conditions",
        
        # SAFE messages - Public service and educational
        "Say no to drugs, and yes to life. Anti Narcotics Cell, Crime Branch, Thane City",
        "Wear your seatbelt for safety. Traffic Police Department",
        "Get vaccinated. Stay safe. Ministry of Health",
        "Report suspicious activity to local police immediately",
        "Fire safety tips: Check smoke detectors monthly",
        "Blood donation camp tomorrow at community center",
        "Environmental awareness: Reduce, reuse, recycle",
        "Vote for your future. Election Commission reminder",
        "Cyber safety: Never share personal information online",
        "Mental health matters. Seek help if needed",
        "Educational workshop on financial literacy next week",
        "Community cleanup drive this Saturday morning",
        "First aid training session registration open",
        "Senior citizen health checkup camp announced",
        "Youth employment program applications accepted",
        "Road safety week: Drive carefully, arrive safely",
        "Water conservation tips for summer season",
        "Digital India initiative: Learn computer skills",
        "Women safety helpline number: Call for help",
        "Child protection awareness: Report abuse cases",
        
        # SAFE messages - Business and notifications
        "Your appointment with Dr. Smith is confirmed for Monday",
        "Grocery store sale: 20% off fresh produce this week",
        "Library closed for maintenance on Sunday",
        "New bus route starting from downtown to airport",
        "Community center yoga classes begin next month",
        "Local farmers market open every Saturday morning",
        "Power outage scheduled for maintenance tomorrow 2-4 PM",
        "Water supply will be interrupted for repairs",
        "Road construction on Main Street, expect delays",
        "New parking regulations effective from next month"
    ],
    "label": [
        # Fraud labels (25 messages)
        "fraud", "fraud", "fraud", "fraud", "fraud", "fraud", "fraud", 
        "fraud", "fraud", "fraud", "fraud", "fraud", "fraud", "fraud",
        "fraud", "fraud", "fraud", "fraud", "fraud", "fraud", "fraud",
        "fraud", "fraud", "fraud", "fraud",
        
        # Safe labels (45 messages) - More safe examples for balance
        "safe", "safe", "safe", "safe", "safe", "safe", "safe",
        "safe", "safe", "safe", "safe", "safe", "safe", "safe", 
        "safe", "safe", "safe", "safe", "safe", "safe", "safe",
        "safe", "safe", "safe", "safe", "safe", "safe", "safe",
        "safe", "safe", "safe", "safe", "safe", "safe", "safe",
        "safe", "safe", "safe", "safe", "safe", "safe", "safe",
        "safe", "safe", "safe", "safe", "safe", "safe", "safe",
        "safe"
    ]
}

df = pd.DataFrame(data)
print(f"Dataset size: {len(df)} messages")
print(f"Fraud messages: {len(df[df['label'] == 'fraud'])}")
print(f"Safe messages: {len(df[df['label'] == 'safe'])}")

# Train model with better parameters and Random Forest for better accuracy
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], 
    test_size=0.2, 
    random_state=42, 
    stratify=df["label"]
)

# Use Random Forest instead of Naive Bayes for better performance
model = make_pipeline(
    TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 3),  # Include trigrams
        min_df=1,
        max_df=0.9,
        max_features=1000
    ), 
    RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
)

model.fit(X_train, y_train)

# Test the model
accuracy = model.score(X_test, y_test)
print(f"Model trained. Accuracy: {accuracy:.2%}")

# Test with specific messages
test_messages = [
    "Say no to drugs, and yes to life. Anti Narcotics Cell, Crime Branch, Thane City",
    "Free Msg-J.P. Morgan Chase Bank Alert-Did You Attempt A Zelle Payment For The Amount of $5000.00? Reply YES or NO",
    "Meeting at 10 AM tomorrow",
    "You won $1000 lottery! Click here to claim"
]

print("\n--- Testing Messages ---")
for msg in test_messages:
    prediction = model.predict([msg])[0]
    confidence = max(model.predict_proba([msg])[0])
    print(f"Message: {msg[:50]}...")
    print(f"Prediction: {prediction} (Confidence: {confidence:.1%})")
    print()

# Save model
joblib.dump(model, "fraud_detector.pkl")
print("✅ Enhanced model saved as fraud_detector.pkl")
