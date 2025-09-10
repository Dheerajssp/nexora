import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os

class CropDiseaseDetector:
    def _init_(self):
        self.model = None
        self.class_names = [
            'Healthy', 'Bacterial_Blight', 'Brown_Spot', 'Leaf_Blast',
            'Tungro', 'Bacterial_Leaf_Streak', 'Sheath_Rot'
        ]
        self.img_size = (224, 224)
    
    def create_model(self, num_classes):
        """Create CNN model for disease detection"""
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.array(image_path)
        
        # Resize image
        image = cv2.resize(image, self.img_size)
        
        # Normalize pixel values
        image = image.astype('float32') / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def detect_disease(self, image_path):
        """Detect disease in crop image"""
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        
        # Predict
        predictions = self.model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Get disease info
        disease = self.class_names[predicted_class]
        
        return {
            'disease': disease,
            'confidence': float(confidence),
            'severity': self.calculate_severity(confidence),
            'treatment': self.get_treatment_recommendation(disease),
            'prevention': self.get_prevention_tips(disease)
        }
    
    def calculate_severity(self, confidence):
        """Calculate disease severity based on confidence"""
        if confidence > 0.9:
            return 'High'
        elif confidence > 0.7:
            return 'Medium'
        elif confidence > 0.5:
            return 'Low'
        else:
            return 'Uncertain'
    
    def get_treatment_recommendation(self, disease):
        """Get treatment recommendations for detected disease"""
        treatments = {
            'Bacterial_Blight': {
                'immediate': 'Apply copper-based fungicide',
                'follow_up': 'Improve field drainage, use resistant varieties',
                'organic': 'Neem oil spray, proper crop rotation'
            },
            'Brown_Spot': {
                'immediate': 'Apply mancozeb or propiconazole',
                'follow_up': 'Balanced fertilization, avoid over-irrigation',
                'organic': 'Baking soda solution, compost application'
            },
            'Leaf_Blast': {
                'immediate': 'Apply tricyclazole or carbendazim',
                'follow_up': 'Proper plant spacing, nitrogen management',
                'organic': 'Trichoderma application, resistant varieties'
            },
            'Healthy': {
                'immediate': 'Continue current practices',
                'follow_up': 'Regular monitoring and preventive care',
                'organic': 'Maintain soil health with organic matter'
            }
        }
        
        return treatments.get(disease, treatments['Healthy'])
    
    def get_prevention_tips(self, disease):
        """Get prevention tips for diseases"""
        prevention = {
            'Bacterial_Blight': [
                'Use certified disease-free seeds',
                'Avoid working in wet fields',
                'Practice crop rotation',
                'Maintain proper plant spacing'
            ],
            'Brown_Spot': [
                'Ensure adequate potassium nutrition',
                'Avoid water stress',
                'Remove crop residues',
                'Use resistant varieties'
            ],
            'Leaf_Blast': [
                'Avoid excessive nitrogen fertilization',
                'Maintain proper water management',
                'Use silicon fertilizers',
                'Plant resistant varieties'
            ]
        }
        
        return prevention.get(disease, ['Regular field monitoring', 'Maintain good agricultural practices'])
    
    def batch_disease_detection(self, image_folder):
        """Process multiple images for disease detection"""
        results = []
        
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_folder, filename)
                result = self.detect_disease(image_path)
                result['filename'] = filename
                results.append(result)
        
        return results
    
    def save_model(self, model_path):
        """Save trained model"""
        self.model.save(f"{model_path}/disease_detection_model.h5")
    
    def load_model(self, model_path):
        """Load trained model"""
        self.model = tf.keras.models.load_model(f"{model_path}/disease_detection_model.h5")

# Usage Example
if _name_ == "_main_":
    detector = CropDiseaseDetector()
    
    # Load pre-trained model (you need to train/download this)
    # detector.load_model('models_trained/disease_detection')
    
    # Test disease detection
    # result = detector.detect_disease('test_images/diseased_leaf.jpg')
    # print("Disease Detection Result:", result)