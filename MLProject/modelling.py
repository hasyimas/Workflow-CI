import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
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
    try:
        X_train, X_test, y_train, y_test = load_data()
    except Exception as e:
        print(f"Error loading data: {e}. Please ensure preprocessing is done.")
        return
    
    
    mlflow.set_experiment("Breast_Cancer_CI_Project")

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    
    with mlflow.start_run():
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)
        
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, average='macro')
        rec = recall_score(y_test, predictions, average='macro')
        f1 = f1_score(y_test, predictions, average='macro')
        
        # Manual Logging
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        mlflow.sklearn.log_model(best_model, "random_forest_tuned")
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Test Accuracy: {acc:.4f}")
        print("Model and metrics logged successfully to DagsHub!")

if __name__ == "__main__":
    main()
