import pickle
import numpy as np

def predict_phishing(sample_input):
    # Load the model and scaler
    with open("/Users/udayanmishra/Desktop/minorproj/spam-message-detection-with-python-and-django/Phishing-Email-Detection-Using-Machine-Learning/model/phishing_model.pkl", "rb") as f:
        model, scaler = pickle.load(f)

    # Ensure input is a 2D numpy array
    sample_input = np.array(sample_input).reshape(1, -1)

    # Scale the input
    sample_input_scaled = scaler.transform(sample_input)

    # Make prediction
    prediction = model.predict(sample_input_scaled)

    # Map result to label
    result = "Phishing" if prediction[0] == 1 else "Legitimate"
    return result

if __name__ == "__main__":
    # Ensure this has exactly 48 features
    example_input =[2, 1, 2, 45, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 15, 23, 0, 0, 0, 0, 0.0136363636, 0.0857142857, 0, 1, 1, 0, 0, 0.0090909091, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1]

    result = predict_phishing(example_input)
    print(f"Prediction: {result}")

