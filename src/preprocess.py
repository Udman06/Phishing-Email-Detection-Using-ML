import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_file, output_file):
    # Load the dataset
    data = pd.read_csv(r'/Users/udayanmishra/Desktop/minorproj/spam-message-detection-with-python-and-django/Phishing-Email-Detection-Using-Machine-Learning/data/Phishing_Legitimate_full.csv')

    # Drop ID column
    data = data.drop(columns=["id"])

    # Separate features and labels
    X = data.drop(columns=["CLASS_LABEL"]).values
    y = data["CLASS_LABEL"].values

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    # Save preprocessed data
    with open(output_file, "wb") as f:
        pickle.dump((X_scaled, y, scaler), f)

    print(f"Preprocessed structured data saved to {output_file}")

if __name__ == "__main__":
    preprocess_data(
        "/Users/udayanmishra/Desktop/minorproj/spam-message-detection-with-python-and-django/Phishing-Email-Detection-Using-Machine-Learning/data/Phishing_Legitimate_full.csv",
        "/Users/udayanmishra/Desktop/minorproj/spam-message-detection-with-python-and-django/Phishing-Email-Detection-Using-Machine-Learning/data/preprocessed_data.pkl"
    )
