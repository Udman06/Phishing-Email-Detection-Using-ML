from flask import Flask, request, jsonify # type: ignore
import pickle
import os
from src.utils import predict_emaildet # adjust imports as per your repo

app = Flask(__name__)
modelpath="/Users/udayanmishra/Desktop/minorproj/spam-message-detection-with-python-and-django/Phishing-Email-Detection-Using-Machine-Learning/models/phishing_detector.pkl"
# Load model and vectorizer
model = pickle.load(open(modelpath, "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    email_text = data.get("email", "")
    if not email_text:
        return jsonify({"error": "No email text provided"}), 400
    

    result = model.predict(email_text)[0]

    return jsonify({"prediction": "Phishing" if result == 1 else "Legitimate"})

if __name__ == '__main__':
    app.run(debug=True)