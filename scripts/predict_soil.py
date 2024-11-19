import joblib
import  numpy as np

#load the saved model
model = joblib.load('../data/soil_model.pkl')
print("Model loaded successfully")

#take user input for soil parameters
print("Enter the following soil parameters:")
N = float(input("Nitrogen (N): "))
P = float(input("Phosphorus (P): "))
K = float(input("Potassium (K): "))
temperature = float(input("Temperature (°C): "))
humidity = float(input("Humidity (%): "))
pH = float(input("pH: "))
rainfall = float(input("Rainfall (mm): "))

#create numpy array for prediction
soil_data = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
prediction = model.predict(soil_data)

#display prediction
print(f"The recommend crop for the given soil parameter is: {prediction[0]}")
