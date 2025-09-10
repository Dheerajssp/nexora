import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class IrrigationOptimizer:
    def _init_(self):
        self.model = None
        self.scaler = StandardScaler()
        self.optimal_moisture_levels = {
            'wheat': {'min': 50, 'max': 70},
            'rice': {'min': 80, 'max': 95},
            'corn': {'min': 55, 'max': 75},
            'cotton': {'min': 45, 'max': 65},
            'sugarcane': {'min': 70, 'max': 85}
        }
    
    def create_model(self):
        """Create irrigation prediction model"""
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        return self.model
    
    def prepare_features(self, sensor_data, weather_data, crop_data):
        """Prepare features for irrigation prediction"""
        features = [
            sensor_data['soil_moisture'],
            sensor_data['temperature'],
            sensor_data['humidity'],
            sensor_data['soil_ph'],
            weather_data['temperature'],
            weather_data['humidity'],
            weather_data['rainfall_prediction'],
            weather_data['wind_speed'],
            crop_data['crop_stage'],  # encoded: 1-germination, 2-vegetative, 3-flowering, 4-maturity
            crop_data['days_since_planting'],
            crop_data['crop_type_encoded']  # wheat=1, rice=2, etc.
        ]
        
        return np.array(features).reshape(1, -1)
    
    def predict_irrigation_needs(self, sensor_data, weather_data, crop_data):
        """Predict irrigation requirements"""
        features = self.prepare_features(sensor_data, weather_data, crop_data)
        features_scaled = self.scaler.transform(features)
        
        # Predict water requirement (liters per square meter)
        water_requirement = self.model.predict(features_scaled)[0]
        
        # Get optimal moisture range for crop
        crop_type = crop_data['crop_type']
        optimal_range = self.optimal_moisture_levels.get(crop_type, {'min': 50, 'max': 70})
        
        # Calculate irrigation recommendation
        current_moisture = sensor_data['soil_moisture']
        target_moisture = (optimal_range['min'] + optimal_range['max']) / 2
        
        irrigation_needed = current_moisture < optimal_range['min']
        
        recommendation = {
            'irrigation_needed': irrigation_needed,
            'water_amount': max(0, water_requirement) if irrigation_needed else 0,
            'current_moisture': current_moisture,
            'target_moisture': target_moisture,
            'optimal_range': optimal_range,
            'urgency': self.calculate_urgency(current_moisture, optimal_range),
            'best_irrigation_time': self.get_optimal_irrigation_time(weather_data),
            'duration_minutes': self.calculate_irrigation_duration(water_requirement),
            'next_check': datetime.now() + timedelta(hours=6)
        }
        
        return recommendation
    
    def calculate_urgency(self, current_moisture, optimal_range):
        """Calculate irrigation urgency"""
        if current_moisture < optimal_range['min'] * 0.7:
            return 'Critical'
        elif current_moisture < optimal_range['min'] * 0.8:
            return 'High'
        elif current_moisture < optimal_range['min']:
            return 'Medium'
        else:
            return 'Low'
    
    def get_optimal_irrigation_time(self, weather_data):
        """Determine best time for irrigation"""
        current_hour = datetime.now().hour
        
        # Avoid irrigation during hot hours (11 AM - 4 PM)
        if 6 <= current_hour <= 10:
            return 'Now (Morning - Optimal)'
        elif 17 <= current_hour <= 20:
            return 'Now (Evening - Good)'
        elif current_hour >= 21 or current_hour <= 5:
            return 'Now (Night - Acceptable)'
        else:
            return 'Wait until evening (5-8 PM)'
    
    def calculate_irrigation_duration(self, water_amount):
        """Calculate irrigation duration based on water amount"""
        # Assuming flow rate of 10 liters per minute per square meter
        flow_rate = 10
        duration = max(5, min(60, water_amount / flow_rate))
        return int(duration)
    
    def create_irrigation_schedule(self, field_data, days=7):
        """Create irrigation schedule for next few days"""
        schedule = []
        
        for day in range(days):
            date = datetime.now() + timedelta(days=day)
            
            # Simulate sensor and weather data for each day
            predicted_conditions = self.simulate_future_conditions(day)
            
            irrigation_rec = self.predict_irrigation_needs(
                predicted_conditions['sensors'],
                predicted_conditions['weather'],
                field_data
            )
            
            if irrigation_rec['irrigation_needed']:
                schedule.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'time': '06:00',
                    'water_amount': irrigation_rec['water_amount'],
                    'duration': irrigation_rec['duration_minutes'],
                    'reason': f"Soil moisture below optimal ({irrigation_rec['current_moisture']:.1f}%)"
                })
        
        return schedule
    
    def simulate_future_conditions(self, days_ahead):
        """Simulate future sensor and weather conditions"""
        # This would integrate with actual weather APIs
        base_temp = 28 + np.random.normal(0, 3)
        base_humidity = 65 + np.random.normal(0, 10)
        
        return {
            'sensors': {
                'soil_moisture': max(20, 60 - days_ahead * 5 + np.random.normal(0, 5)),
                'temperature': base_temp,
                'humidity': base_humidity,
                'soil_ph': 6.8
            },
            'weather': {
                'temperature': base_temp,
                'humidity': base_humidity,
                'rainfall_prediction': np.random.exponential(2),
                'wind_speed': 5 + np.random.normal(0, 2)
            }
        }
    
    def optimize_water_usage(self, field_data, water_budget):
        """Optimize water usage within budget constraints"""
        total_area = field_data['area_hectares']
        
        # Calculate water efficiency score
        efficiency_factors = {
            'drip_irrigation': 0.9,
            'sprinkler': 0.75,
            'flood': 0.5
        }
        
        irrigation_method = field_data.get('irrigation_method', 'sprinkler')
        efficiency = efficiency_factors.get(irrigation_method, 0.75)
        
        effective_water_budget = water_budget * efficiency
        
        recommendation = {
            'recommended_method': 'drip_irrigation' if water_budget < 1000 else irrigation_method,
            'water_savings_potential': water_budget * (0.9 - efficiency) if efficiency < 0.9 else 0,
            'cost_savings': self.calculate_cost_savings(water_budget, efficiency),
            'efficiency_score': efficiency * 100
        }
        
        return recommendation
    
    def calculate_cost_savings(self, water_amount, efficiency):
        """Calculate potential cost savings from efficient irrigation"""
        water_cost_per_liter = 0.02  # ₹0.02 per liter
        savings = water_amount * water_cost_per_liter * (0.9 - efficiency)
        return max(0, savings)
    
    def save_model(self, model_path):
        """Save irrigation model"""
        joblib.dump(self.model, f"{model_path}/irrigation_model.pkl")
        joblib.dump(self.scaler, f"{model_path}/irrigation_scaler.pkl")
    
    def load_model(self, model_path):
        """Load irrigation model"""
        self.model = joblib.load(f"{model_path}/irrigation_model.pkl")
        self.scaler = joblib.load(f"{model_path}/irrigation_scaler.pkl")

# Usage Example
if _name_ == "_main_":
    optimizer = IrrigationOptimizer()
    optimizer.create_model()
    
    # Sample data
    sensor_data = {
        'soil_moisture': 35,
        'temperature': 28,
        'humidity': 60,
        'soil_ph': 6.8
    }
    
    weather_data = {
        'temperature': 32,
        'humidity': 55,
        'rainfall_prediction': 0,
        'wind_speed': 8
    }
    
    crop_data = {
        'crop_type': 'wheat',
        'crop_stage': 2,
        'days_since_planting': 45,
        'crop_type_encoded': 1
    }
    
    # Get irrigation recommendation
    # recommendation = optimizer.predict_irrigation_needs(sensor_data, weather_data, crop_data)
    # print("Irrigation Recommendation:", recommendation)