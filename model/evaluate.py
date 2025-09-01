import pandas as pd
from sklearn.metrics import silhouette_score
import joblib

# Load the dataset
data = pd.read_csv('data/Mall_Customers.csv')

# Preprocess the dataset
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Load the saved scaler and model
scaler = joblib.load('model/scaler.pkl')
model = joblib.load('model/customer_segmentation.pkl')

# Transform the data
X_scaled = scaler.transform(X)

# Make predictions
y_pred = model.predict(X_scaled)

# Evaluate the model
silhouette = silhouette_score(X_scaled, y_pred)
print(f'Model silhouette score: {silhouette:.2f}')
