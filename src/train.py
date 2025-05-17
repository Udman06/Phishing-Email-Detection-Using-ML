import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

def train_model(preprocessed_data_file, model_output_file):
    # Load the preprocessed data
    with open(preprocessed_data_file, "rb") as f:
        X, y, scaler = pickle.load(f)

    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the model
    os.makedirs(os.path.dirname(model_output_file), exist_ok=True)
    with open(model_output_file, "wb") as f:
        pickle.dump((model, scaler), f)

    print(f"Trained model saved to {model_output_file}")

if __name__ == "__main__":
    train_model(
        "/Users/udayanmishra/Desktop/minorproj/spam-message-detection-with-python-and-django/Phishing-Email-Detection-Using-Machine-Learning/data/preprocessed_data.pkl",
        "/Users/udayanmishra/Desktop/minorproj/spam-message-detection-with-python-and-django/Phishing-Email-Detection-Using-Machine-Learning/model/phishing_model.pkl"
    )
