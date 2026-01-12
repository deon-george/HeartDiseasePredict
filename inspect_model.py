import joblib
import pandas as pd
import os

try:
    model = joblib.load("model.pkl")
    print(f"Model Type: {type(model)}")
    
    if hasattr(model, 'coef_'):
        print("Model has coefficients:")
        print(model.coef_)
    
    if hasattr(model, 'feature_importances_'):
        print("Model has feature importances:")
        print(model.feature_importances_)
        
    print(f"Classes: {model.classes_}")
except Exception as e:
    print(f"Error inspecting model: {e}")
