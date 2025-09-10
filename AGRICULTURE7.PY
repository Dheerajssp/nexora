import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

class CropPredictionModel:
    def _init_(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = [
            'temperature', 'humidity', 'ph', 'rainfall', 
            'nitrogen', 'phosphorus', 'potassium'
        ]
    
    def create_model(self, input_shape, num_classes):
        """Create deep learning model for crop prediction"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, data_path):
        """Train the crop prediction model"""
        # Load and preprocess data
        df = pd.read_csv(data_path)
        
        # Features and target
        X = df[self.feature_names]
        y = df['crop']
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42
        )
        
        # Create and train model
        self.model = self.create_model(X_train.shape[1], len(np.unique(y_encoded)))
        
        # Train with callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return history
    
    def predict_crop(self, soil_data):
        """Predict best crop for given soil conditions"""
        # Prepare input data
        input_data = np.array([[
            soil_data['temperature'],
            soil_data['humidity'],
            soil_data['ph'],
            soil_data['rainfall'],
            soil_data['nitrogen'],
            soil_data['phosphorus'],
            soil_data['potassium']
        ]])
        
        # Scale input
        input_scaled = self.scaler.transform(input_data)
        
        # Predict
        predictions = self.model.predict(input_scaled)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Decode prediction
        predicted_crop = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return {
            'recommended_crop': predicted_crop,
            'confidence': float(confidence),
            'all_predictions': {
                crop: float(prob) 
                for crop, prob in zip(
                    self.label_encoder.classes_, 
                    predictions[0]
                )
            }
        }
    
    def get_crop_recommendations(self, soil_data, top_n=3):
        """Get top N crop recommendations"""
        input_data = np.array([[
            soil_data['temperature'],
            soil_data['humidity'],
            soil_data['ph'],
            soil_data['rainfall'],
            soil_data['nitrogen'],
            soil_data['phosphorus'],
            soil_data['potassium']
        ]])
        
        input_scaled = self.scaler.transform(input_data)
        predictions = self.model.predict(input_scaled)[0]
        
        # Get top predictions
        top_indices = np.argsort(predictions)[-top_n:][::-1]
        
        recommendations = []
        for idx in top_indices:
            crop = self.label_encoder.inverse_transform([idx])[0]
            confidence = predictions[idx]
            
            recommendations.append({
                'crop': crop,
                'confidence': float(confidence),
                'suitability_score': float(confidence * 100)
            })
        
        return recommendations
    
    def save_model(self, model_path):
        """Save trained model and preprocessors"""
        self.model.save(f"{model_path}/crop_model.h5")
        joblib.dump(self.scaler, f"{model_path}/crop_scaler.pkl")
        joblib.dump(self.label_encoder, f"{model_path}/crop_encoder.pkl")
    
    def load_model(self, model_path):
        """Load trained model and preprocessors"""
        self.model = tf.keras.models.load_model(f"{model_path}/crop_model.h5")
        self.scaler = joblib.load(f"{model_path}/crop_scaler.pkl")
        self.label_encoder = joblib.load(f"{model_path}/crop_encoder.pkl")

# Usage Example
if _name_ == "_main_":
    crop_model = CropPredictionModel()
    
    # Train model (uncomment when you have training data)
    # crop_model.train_model('data/crop_data.csv')
    # crop_model.save_model('models_trained/crop_prediction')
    
    # Load pre-trained model
    # crop_model.load_model('models_trained/crop_prediction')
    
    # Test prediction
    sample_soil = {
        'temperature': 25.5,
        'humidity': 65.0,
        'ph': 6.8,
        'rainfall': 150.0,
        'nitrogen': 40,
        'phosphorus': 25,
        'potassium': 35
    }
    
    # result = crop_model.predict_crop(sample_soil)
    # print("Crop Prediction:", result)