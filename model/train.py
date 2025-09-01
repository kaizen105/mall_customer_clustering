import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

# Load the dataset
data = pd.read_csv('data/Mall_Customers.csv')

# Select relevant features (Income & Spending Score)
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale the features (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a KMeans clustering model
kmeans = KMeans(n_clusters=5, random_state=42)  # you can tune n_clusters
kmeans.fit(X_scaled)

# Save the scaler and model
os.makedirs('model', exist_ok=True)
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(kmeans, 'model/customer_segmentation.pkl')

print("✅ Model training complete. Files saved in 'model/'")
