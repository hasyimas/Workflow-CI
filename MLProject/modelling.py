import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'breast_cancer_preprocessing')
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).squeeze()
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).squeeze()
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = load_data()

    # DagsHub Tracking (To be populated by Github Actions or user)
    # DAGSHUB_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    # mlflow.set_tracking_uri(DAGSHUB_TRACKING_URI)
    
    mlflow.set_experiment("Breast_Cancer_CI_Project")
    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"Model accuracy: {acc:.4f}")
        
        # Saving model explicitly
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()
