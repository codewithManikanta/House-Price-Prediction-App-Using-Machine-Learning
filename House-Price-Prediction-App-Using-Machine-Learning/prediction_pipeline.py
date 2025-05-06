import numpy as np

class PredictionPipeline:
    def __init__(self, location_encoder, scaler, model):
        self.location_encoder = location_encoder
        self.scaler = scaler
        self.model = model
    
    def predict(self, X):
        # Encode location
        location_encoded = self.location_encoder.transform(X[['location']])
        # Convert other features
        other_features = X[['total_sqft', 'bath', 'balcony', 'bedrooms']].to_numpy()
        # Combine features
        X_encoded = np.hstack([location_encoded, other_features])
        # Scale features
        X_scaled = self.scaler.transform(X_encoded)
        # Make prediction
        return self.model.predict(X_scaled)