# train_model.py

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import numpy as np  # Added import for numpy

# Generate random features and labels for testing
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = np.random.randint(0, 2, size=100)  # Binary labels (0 or 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a machine learning model (replace 'YourModel' with your actual model)
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance (replace 'accuracy' with your preferred metric)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model to a file (replace 'your_model_filename.pkl' with your desired filename)
joblib.dump(model, 'your_model_filename.pkl')
print("Model saved successfully!")
