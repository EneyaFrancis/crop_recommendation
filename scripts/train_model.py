import pandas as pd
from scipy.special import y_pred
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

#load dataset
data = pd.read_csv('../data/Crop_recomendation.csv')

#explore the data
print("First 5 rows of the dataset")
print(data.head())

#split features (X) and targets (Y)
X = data.drop('label', axis=1)
y = data['label']

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train a random forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

#evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

#save the trained model
joblib.dump(model, '../data/soil_model.pkl')
print("Model saved as 'soil model.pkl'")