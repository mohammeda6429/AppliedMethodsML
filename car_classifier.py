import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load the data
data = np.loadtxt('Hwk1.txt', delimiter=',', skiprows=1, dtype='str')

X = data[:, :-1].astype(float)
y = data[:, -1]  # Labels: car type

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train the decision tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y_encoded)

# Predict the car type for a vehicle with 250 horsepower and 3 seats
new_vehicle = np.array([[250, 3]])
prediction_encoded = clf.predict(new_vehicle)
prediction = label_encoder.inverse_transform(prediction_encoded)

# Output
print(f"The vehicle with 250 horsepower and 3 seats is predicted to be a {prediction[0]}")
