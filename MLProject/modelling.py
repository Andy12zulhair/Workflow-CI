import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def main():
    # Load dataset
    data_path = os.path.join("diabetes_risk_preprocessing", "diabetes_risk_clean.csv")
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    target_col = 'diabetes_risk'
    if target_col not in df.columns:
        target_col = df.columns[-1]
        
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # We will log everything locally to a saved_model directory
    mlflow.autolog(disable=True)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model trained. Accuracy: {acc:.4f}")
    
    # Save the model artifact explicitly to local directory for Docker build
    model_dir = "saved_model"
    if os.path.exists(model_dir):
        import shutil
        shutil.rmtree(model_dir)
        
    mlflow.sklearn.save_model(
        model, 
        model_dir, 
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
    )
    print(f"Model saved locally to {model_dir}")

if __name__ == "__main__":
    main()
