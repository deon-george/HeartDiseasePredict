import numpy as np
import pandas as pd
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(BASE_DIR, "heart.csv"))
data=data.fillna(data.mode(numeric_only=True))
#data=data.drop(["restecg","slope","thal"],axis=1)
#print(data.head())
from sklearn.impute import SimpleImputer
imputer= SimpleImputer(strategy="mean")
from sklearn.linear_model import LogisticRegression
np.random.seed(42)
shuffled_indices = np.random.permutation(len(data))
test_size = int(0.2 * len(data))
test_indices = shuffled_indices[:test_size]
train_indices = shuffled_indices[test_size:]
train_set = data.iloc[train_indices]
test_set = data.iloc[test_indices]
X_train = train_set.drop("target", axis=1)
y_train = train_set["target"]
X_test = test_set.drop("target", axis=1)
y_test = test_set["target"]
X_train=imputer.fit_transform(X_train)
X_test=imputer.transform(X_test)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("accuracy of the model is",accuracy)
import joblib
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
imputer_path = os.path.join(BASE_DIR, "imputer.pkl")
joblib.dump(clf, model_path)
joblib.dump(imputer, imputer_path)
print("✅ Files saved at:")
print(model_path)
print(imputer_path)
import joblib



